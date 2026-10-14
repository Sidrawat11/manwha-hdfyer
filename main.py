import os
# Must be set before torch is imported: lets the allocator grow segments instead of fragmenting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import sys
import multiprocessing as mp
import queue
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from pathlib import Path
from core import extractor
from core.model_loader import load_student
from core.packager import Packager
from pipeline.page_buffer import PageBuffer
from pipeline.finisher import PageFinisher
from inference.chunker import ChunkMeta, chunk_page
from inference import engine, trt_backend
from inference.batcher import build_batches
from inference.engine import BACKEND, BACKENDS, engine_stats, prepare_chapter, select_backend, upscale_stream
from profiler.hdfyer import BatchProfile

cudnn.benchmark = True

# Production runs one model: hdfyer, our native 2x upscaler (SPAN S, distilled in experiments/exp16).
# The teacher it was distilled from (4x-UltraSharp) is only reachable from experiments and tests.
WEIGHTS = Path("models/hdfyer/hdfyer-2x-s-r2.pth")
SCALE = 2
STRIP_HEIGHT = 512   # input rows; every strip of every page is exactly this tall (see inference/chunker.py)
BATCH = "auto"       # strips per forward pass: "auto" reads the GPU's profile (profiler/hdfyer.py), or a number

# Teacher only. True restores the old strips: tallest strip that fits VRAM, 128-row overlap, short last strip.
LEGACY_STRIPS = False

# CPU finishing (sharpen, cleanup, encode) runs on threads while the GPU works on the next pages.
CPU_WORKERS = min(4, os.cpu_count() or 1)
MAX_INFLIGHT = 3   # stitched pages queued or being finished at once; a 2x page is ~45 MB (a teacher 4x page ~180 MB)

# The next chapter is read and decoded in the background (on this many threads) while the current one is upscaled.
DECODE_THREADS = 4
_decode_threads = DECODE_THREADS   # what this process uses: DECODE_THREADS, or less with --parallel (configure_threads)

# Chapters at once, each in its own process with its own copy of the model: feeds a big GPU that one Python thread can't.
PARALLEL = 1
PARALLEL_MIN_VRAM_GB = 16   # below this one pipeline already keeps the GPU busy


def run_teacher_profiler():
    """Profiles the teacher's VRAM use and writes its benchmark map. Experiments only: the production run never calls this."""
    from profiler.cache import cache_check, save_cache
    from profiler.scanner import scan_library
    from profiler.runner import run_profiler
    from profiler.benchmark import build_map, get_lookup
    if cache_check():
        print("Cache found, skipping profiling.")
        return

    library_dir = Path("Manwhas")
    model_path = Path("models/4x-UltraSharp.pth")

    print("Scanning library...")
    unique_widths, max_height = scan_library(library_dir)
    print(f"Widths found: {unique_widths}, Max height: {max_height}")

    print("Running profiler...")
    raw_results = run_profiler(unique_widths, max_height, model_path)

    print("Building benchmark map...")
    benchmark_map = build_map(raw_results)
    save_cache(benchmark_map)
    print("Cache saved.")

    print("Test lookups:")
    test_cases = [(720, 512), (720, 800), (720, 3200), (800, 1024)]
    for w, h in test_cases:
        result = get_lookup(w, h, benchmark_map)
        print(f"  {w}x{h} -> {result}")


def process_chapter_v2(cbz_path: Path, model: torch.nn.Module, benchmark_map: dict, output_path: Path, scale: int = 4, legacy_strips: bool = LEGACY_STRIPS,
                       cpu_workers: int = CPU_WORKERS, max_inflight: int = MAX_INFLIGHT, overlap: bool = True,
                       strip_height: int | None = None, batch_size=None, quiet: bool = False, pages=None):
    """
    Upscales one chapter with whatever model the caller loaded. `scale` is that model's output scale (4 or 2);
    the CBZ is 2x either way. hdfyer passes strip_height and batch_size (a number, or a function of strip height
    and width) and no benchmark map; the defaults are the teacher's, for experiments.
    quiet drops the per-chapter progress output (--parallel workers); fallbacks are still printed.
    pages: the chapter's (name, image) list if it was already loaded (see prefetched); otherwise it is loaded here.
    """
    log = (lambda *args, **kwargs: None) if quiet else print
    log(f"Loading pages from {cbz_path}...")
    loaded = extractor.load_chapter(cbz_path, _decode_threads) if pages is None else pages
    pages = [image for _, image in loaded]
    page_names = {page_id: name for page_id, (name, _) in enumerate(loaded)}
    del loaded
    log(f"Loaded {len(pages)} pages.")

    log("Chunking pages...")
    all_chunks = []
    for page_id, image in enumerate(pages):
        page_chunks = chunk_page(image, page_id, benchmark_map, legacy_strips=legacy_strips, strip_height=strip_height)
        for chunk in page_chunks:
            chunk.page_name = page_names[page_id]   # so a fallback can say which page it happened on
        all_chunks.extend(page_chunks)
    log(f"Total chunks: {len(all_chunks)}")

    batches = build_batches(all_chunks, benchmark_map, batch_size)
    log(f"Total batches: {len(batches)}")

    # Builds any missing TensorRT engine now, so the one-time cost isn't hidden inside the progress bar.
    prepare_chapter(all_chunks, scale)

    page_buffer = PageBuffer(scale)
    # A worker's exception surfaces in submit() or when the finisher closes, and fails the chapter.
    with Packager(output_path, 'jpg', 92) as packager, PageFinisher(packager, cpu_workers, max_inflight, overlap, scale) as finisher:
        # One batch in flight: stitching and hand-off below run while the GPU works on the next batch.
        for batch, results in tqdm(upscale_stream(batches, model, scale), total=len(batches), desc="Upscaling batches", disable=quiet):
            for upscaled_array, meta in results:
                completed_page = page_buffer.add(upscaled_array, meta)
                if completed_page is not None:
                    finisher.submit(page_names[meta.page_id], completed_page)
                    del completed_page

    stats = engine_stats()
    if stats["switches"]:
        log(f"TensorRT: {stats['switches']} engine switches, {sum(stats['load_seconds']):.1f}s loading engines.")
    log(f"Output written to {output_path}")


def parse_batch(text: str) -> str | int:
    """--batch: "auto" or a whole number of strips, at least 1."""
    if text == "auto":
        return text
    try:
        batch = int(text)
    except ValueError:
        batch = 0
    if batch < 1:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a number of at least 1, got {text!r}")
    return batch


def print_run_summary(backend: str, builds: list[float] | None = None, events: list[dict] | None = None) -> None:
    """End-of-run report: engines built, and every fallback from the selected backend. Defaults to this process's own."""
    builds = engine_stats()["build_seconds"] if builds is None else builds
    events = engine.fallback_events() if events is None else events
    if builds:
        print(f"TensorRT engines built this run: {len(builds)}, {sum(builds):.0f}s in total.")
    if not events:
        print(f"Fallbacks: none. Every strip ran on {backend}.")
        return
    print(f"Fallbacks: {len(events)}")
    for event in events:
        print("  " + engine.format_event(event))


def load_runtime(weights: Path, backend: str, batch: str | int, allow_build: bool = True, measure: bool = True):
    """
    Loads hdfyer and selects its backend. Returns (model, backend in use, batch size or BatchProfile).
    allow_build=False and measure=False are for --parallel workers: they use what the warm-up built and measured,
    and anything missing falls back loudly instead of being built or measured.
    """
    model = load_student(weights)

    # auto means TensorRT, or eager if TensorRT isn't installed. Nothing here can reach another model.
    requested = backend
    if backend == "auto":
        backend = "trt" if trt_backend.is_available() else "eager"
    batch_size = batch
    if batch == "auto":
        weights_id = f"{weights.name}-{trt_backend.file_sha256(weights)[:trt_backend.HASH_LENGTH]}"
        on_missing = None
        if not measure:
            def on_missing(width):
                engine.report_fallback("no batch profile", f"{width} px wide pages were not profiled in the warm-up; using batch 1")
                return 1
        batch_size = BatchProfile(model, weights_id, STRIP_HEIGHT, before_measuring=engine.free_gpu, on_missing=on_missing)
    backend = select_backend(backend, model, weights, None, strip_height=STRIP_HEIGHT, round_output=True,
                             trt_batch=batch_size, arch="span")
    if engine._runtime.trt is not None:
        engine._runtime.trt.allow_build = allow_build
    if requested == "auto" and backend != "trt":
        engine.report_fallback("backend unavailable", f"TensorRT is not installed (pip install tensorrt-cu12 onnx); using {backend}")
    return model, backend, batch_size


def configure_threads(parallel: int) -> dict:
    """
    Caps this process's thread pools at its share of the CPUs, max(1, cpu_count // parallel): torch's intra-op
    threads, OpenCV's threads and the decode pool. Every pool is only ever lowered (min with what it is now), so a
    single process (parallel=1) keeps the defaults. Without this, N processes each start a pool sized for the whole
    machine (torch: one thread per core) and the threads of all of them fight over the same cores.
    """
    global _decode_threads
    share = max(1, (os.cpu_count() or 1) // parallel)
    torch.set_num_threads(min(torch.get_num_threads(), share))
    cv2.setNumThreads(min(cv2.getNumThreads(), share))
    _decode_threads = min(DECODE_THREADS, share)
    return {"share": share, "torch": torch.get_num_threads(), "opencv": cv2.getNumThreads(), "decode": _decode_threads}


def prefetched(chapters, load_threads: int | None = None):
    """
    Yields ((cbz, relative), pages) for each chapter, reading and decoding the NEXT chapter in a background thread
    while the caller works on this one, so the GPU never waits for JPEG/PNG decoding. One chapter ahead: at most
    two decoded chapters are held at once. A chapter that fails to load raises when its turn comes.
    """
    chapters = iter(chapters)
    load_threads = load_threads or _decode_threads
    with ThreadPoolExecutor(1, thread_name_prefix="prefetch") as reader:
        load = lambda chapter: extractor.load_chapter(chapter[0], load_threads)
        current = next(chapters, None)
        pending = reader.submit(load, current) if current is not None else None
        while current is not None:
            following = next(chapters, None)
            pages = pending.result()
            pending = reader.submit(load, following) if following is not None else None
            yield current, pages
            del pages
            current = following


def todo_chapters(manwha_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """(cbz, relative path) of every chapter without a finished output. A leftover .part does not count as finished."""
    todo = []
    for cbz_path in sorted(manwha_dir.rglob("*.cbz")):
        relative = cbz_path.relative_to(manwha_dir)
        if (output_dir / relative).exists():
            tqdm.write(f"Skipping, already processed: {relative}")
            continue
        todo.append((cbz_path, relative))
    return todo


def process_chapter(cbz_path: Path, relative: Path, output_dir: Path, model, batch_size, cpu_workers: int, max_inflight: int,
                    overlap: bool, quiet: bool = False, pages=None) -> float:
    """One chapter into output_dir/relative. Returns its time in seconds."""
    output_path = output_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to .part and rename on success, so an interrupted chapter isn't mistaken for a finished one on resume.
    chapter_start = time.perf_counter()
    part_path = output_path.with_name(output_path.name + ".part")
    process_chapter_v2(cbz_path, model, None, part_path, SCALE, False, cpu_workers, max_inflight, overlap,
                       STRIP_HEIGHT, batch_size, quiet=quiet, pages=pages)
    part_path.replace(output_path)
    torch.cuda.empty_cache()
    return time.perf_counter() - chapter_start


def process_all(manwha_dir: Path, output_dir: Path, weights: Path = WEIGHTS, backend: str = BACKEND, batch: str | int = BATCH,
                cpu_workers: int = CPU_WORKERS, max_inflight: int = MAX_INFLIGHT, overlap: bool = True, parallel: int = PARALLEL):
    """Upscales the library with hdfyer, one chapter at a time, or `parallel` chapters at once in worker processes."""
    if parallel > 1:
        return process_all_parallel(manwha_dir, output_dir, weights, backend, batch, cpu_workers, max_inflight, overlap, parallel)

    print(f"Loading hdfyer ({weights})...")
    configure_threads(1)
    model, backend, batch_size = load_runtime(weights, backend, batch)

    cbz_files = sorted(manwha_dir.rglob("*.cbz"))
    print(f"Found {len(cbz_files)} chapters.")

    total_start = time.perf_counter()
    completed = 0

    try:
        todo = todo_chapters(manwha_dir, output_dir)
        for (cbz_path, relative), pages in tqdm(prefetched(todo), total=len(todo), desc="Processing chapters"):
            engine.set_context(str(relative))
            elapsed = process_chapter(cbz_path, relative, output_dir, model, batch_size, cpu_workers, max_inflight, overlap,
                                      pages=pages)
            del pages
            completed += 1
            tqdm.write(f"Done: {relative} in {elapsed:.1f}s")
    finally:
        # Also after a failure or Ctrl+C, so fallbacks that led up to it are not lost.
        total_elapsed = time.perf_counter() - total_start
        avg = total_elapsed / completed if completed else 0
        print(f"\nFinished {completed} chapters in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min), avg {avg:.1f}s/chapter.")
        print_run_summary(backend)


# ---------------------------------------------------------------- --parallel N

def _warm_up(weights: Path, backend: str, batch: str | int, cbz_paths: list[str], results) -> None:
    """
    Warm-up process, run before any worker: measures every missing batch-profile entry and builds every missing
    TensorRT engine for the page widths in the to-do chapters, so that no worker ever writes a cache file.
    """
    try:
        configure_threads(1)   # runs alone, before any worker
        model, backend_used, batch_size = load_runtime(weights, backend, batch)
        engine.set_context("warm-up")
        widths = sorted({w for path in cbz_paths for w, _ in extractor.page_sizes(path)})
        if callable(batch_size):
            for width in widths:
                batch_size(STRIP_HEIGHT, width)
        # One standard strip per width: prepare_chapter builds whatever engine is missing, shrinking the batch on failure.
        strips = [ChunkMeta(0, np.zeros((STRIP_HEIGHT, width, 3), np.uint8), 0, 1, 0, STRIP_HEIGHT, 0, 0,
                            page_name=f"{width} px wide pages") for width in widths]
        engine.prepare_chapter(strips, SCALE)
        results.put({"backend": backend_used, "widths": widths, "builds": engine_stats()["build_seconds"],
                     "events": engine.fallback_events(), "state": engine.warm_up_state(),
                     "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3})
    except BaseException:
        results.put({"error": traceback.format_exc()})


def _worker(worker_id: int, weights: Path, backend: str, batch: str | int, state: dict, output_dir: Path,
            cpu_workers: int, max_inflight: int, overlap: bool, parallel: int, tasks, results) -> None:
    """One pipeline: loads hdfyer once, then processes chapters from `tasks` until it gets None."""
    # Ctrl+C is the parent's to handle: it stops the workers, so a half-done chapter is simply left as .part.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        threads = configure_threads(parallel)
        model, _, batch_size = load_runtime(weights, backend, batch, allow_build=False, measure=False)
        engine.adopt_warm_up(state["failed_shapes"], state["batch_override"])
        reported = len(engine.fallback_events())
        results.put({"type": "ready", "worker": worker_id, "events": engine.fallback_events(), "threads": threads})
        def queued():
            while (task := tasks.get()) is not None:
                yield Path(task[0]), Path(task[1])

        for (cbz_path, relative), pages in prefetched(queued()):
            engine.set_context(f"w{worker_id} {relative}")
            elapsed = process_chapter(cbz_path, relative, output_dir, model, batch_size, cpu_workers, max_inflight, overlap,
                                      quiet=True, pages=pages)
            del pages
            events = engine.fallback_events()
            results.put({"type": "done", "worker": worker_id, "relative": str(relative), "seconds": elapsed,
                         "events": events[reported:]})
            reported = len(events)
        results.put({"type": "exit", "worker": worker_id, "events": engine.fallback_events()[reported:],
                     "builds": engine_stats()["build_seconds"]})
    except BaseException:
        results.put({"type": "error", "worker": worker_id, "error": traceback.format_exc()})


def process_all_parallel(manwha_dir: Path, output_dir: Path, weights: Path, backend: str, batch: str | int,
                         cpu_workers: int, max_inflight: int, overlap: bool, parallel: int) -> None:
    """
    `parallel` worker processes share a queue of to-do chapters. A single warm-up process first fills the engine
    and batch-profile caches for every page width in those chapters, so the workers only ever read them.
    The parent process never touches the GPU.
    """
    ctx = mp.get_context("spawn")
    todo = todo_chapters(manwha_dir, output_dir)
    print(f"{len(todo)} chapters to do, {parallel} worker processes.")
    if not todo:
        return

    print("Warm-up: batch profiles and TensorRT engines for every page width in those chapters...")
    warm_results = ctx.Queue()
    warm = ctx.Process(target=_warm_up, args=(weights, backend, batch, [str(cbz) for cbz, _ in todo], warm_results))
    warm.start()
    try:
        warm_up = warm_results.get()
    finally:
        if warm.is_alive() and "warm_up" not in locals():
            warm.terminate()   # Ctrl+C during the warm-up: an engine being built is written to a private temp file, never half-saved
        warm.join()
    if "error" in warm_up:
        raise RuntimeError("warm-up failed:\n" + warm_up["error"])
    backend = warm_up["backend"]
    print(f"Warm-up done: widths {warm_up['widths']}, backend {backend}.")
    if warm_up["vram_gb"] < PARALLEL_MIN_VRAM_GB:
        print(f"WARNING: --parallel {parallel} on a GPU with {warm_up['vram_gb']:.0f} GB of VRAM. Below {PARALLEL_MIN_VRAM_GB} GB "
              f"one pipeline already keeps the GPU busy, so extra workers mostly compete for it and for memory.")

    tasks, results = ctx.Queue(), ctx.Queue()
    for cbz, relative in todo:
        tasks.put((str(cbz), str(relative)))
    for _ in range(parallel):
        tasks.put(None)
    workers = [ctx.Process(target=_worker, args=(i + 1, weights, backend, batch, warm_up["state"], output_dir,
                                                 cpu_workers, max_inflight, overlap, parallel, tasks, results))
               for i in range(parallel)]

    builds, events = list(warm_up["builds"]), list(warm_up["events"])
    completed, exited = 0, set()
    total_start = time.perf_counter()
    progress = tqdm(total=len(todo), desc="Processing chapters")
    try:
        for worker in workers:
            worker.start()
        while len(exited) < parallel:
            try:
                message = results.get(timeout=2)
            except queue.Empty:
                dead = [i + 1 for i, w in enumerate(workers) if not w.is_alive() and (i + 1) not in exited]
                if dead:
                    raise RuntimeError(f"worker(s) {dead} died without reporting (killed, or out of host memory?)")
                continue
            kind = message["type"]
            if kind == "ready":
                events += message["events"]
                if message["worker"] == 1:
                    t = message["threads"]
                    tqdm.write(f"Threads per worker (share of {os.cpu_count()} CPUs: {t['share']}): torch {t['torch']}, "
                               f"OpenCV {t['opencv']}, decode {t['decode']}, finishing {cpu_workers}.")
            elif kind == "done":
                completed += 1
                events += message["events"]
                progress.update(1)
                tqdm.write(f"[w{message['worker']}] Done: {message['relative']} in {message['seconds']:.1f}s")
            elif kind == "exit":
                exited.add(message["worker"])
                events += message["events"]
                builds += message["builds"]
            elif kind == "error":
                raise RuntimeError(f"worker {message['worker']} failed:\n{message['error']}")
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        tqdm.write("Interrupted: stopping the workers. Chapters they were on stay as .part files and are redone next run.")
        raise
    finally:
        progress.close()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()   # mid-chapter: its output is still a .part, never renamed, so never taken as finished
        for worker in workers:
            if worker.pid is not None:
                worker.join(timeout=30)
        for q in (tasks, results):
            q.cancel_join_thread()   # nobody will drain them now; don't block exit on their feeder threads
            q.close()
        total_elapsed = time.perf_counter() - total_start
        avg = total_elapsed / completed if completed else 0
        print(f"\nFinished {completed} chapters in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min), avg {avg:.1f}s/chapter, "
              f"{parallel} workers.")
        print_run_summary(backend, builds, events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale every chapter in Manwhas/ into results/ with hdfyer.")
    parser.add_argument("--backend", choices=BACKENDS, default=BACKEND,
                        help="inference runtime: auto is TensorRT if installed, else eager (default: %(default)s)")
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="hdfyer weights (default: %(default)s)")
    parser.add_argument("--batch", type=parse_batch, default=BATCH,
                        help="strips per forward pass: auto uses the best batch measured for this GPU and page width, "
                             "profiling it on first use; a number overrides it (default: %(default)s)")
    parser.add_argument("--cpu-workers", type=int, default=CPU_WORKERS,
                        help="threads finishing pages (sharpen, cleanup, encode) while the GPU works (default: %(default)s)")
    parser.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT,
                        help="most upscaled pages queued or being finished at once, ~45 MB each (default: %(default)s)")
    parser.add_argument("--no-overlap", action="store_true",
                        help="finish each page on the main thread before the GPU continues (the old serial path)")
    parser.add_argument("--parallel", type=int, default=PARALLEL,
                        help="chapters processed at once, each in its own worker process; helps on large GPUs "
                             f"(16 GB+), not on laptops (default: %(default)s)")
    args = parser.parse_args()
    if args.cpu_workers < 1 or args.max_inflight < 1 or args.parallel < 1:
        parser.error("--cpu-workers, --max-inflight and --parallel must be at least 1")
    if not args.weights.exists():
        parser.error(f"hdfyer weights not found: {args.weights}")

    try:
        process_all(Path("Manwhas"), Path("results"), args.weights, backend=args.backend, batch=args.batch,
                    cpu_workers=args.cpu_workers, max_inflight=args.max_inflight, overlap=not args.no_overlap, parallel=args.parallel)
    except KeyboardInterrupt:
        # The summary has been printed; unfinished chapters are .part files and are redone on the next run.
        sys.exit(130)
