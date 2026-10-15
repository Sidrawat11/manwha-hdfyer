"""The pipeline: ingest -> digest -> render, one chapter at a time.

process_all walks the library (ingest), and for each chapter process_chapter cuts the pages into
strips and upscales them (digest), then sharpens, encodes and writes the CBZ (render).
With parallel > 1 it hands over to hdfyer.parallel, which runs several of these pipelines at once.
"""

import os
import time
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from pathlib import Path
from hdfyer.ingest import reader
from hdfyer.ingest.library import todo_chapters
from hdfyer.ingest.reader import prefetched
from hdfyer.digest.model import load_student
from hdfyer.digest.stitch import PageBuffer
from hdfyer.digest.strips import chunk_page
from hdfyer.digest import engine, trt as trt_backend
from hdfyer.digest.batches import build_batches
from hdfyer.digest.engine import BACKEND, BACKENDS, engine_stats, prepare_chapter, select_backend, upscale_stream
from hdfyer.digest.batch_profile import BatchProfile
from hdfyer.render.package import Packager
from hdfyer.render.finish import PageFinisher

cudnn.benchmark = True

# Production runs one model: hdfyer, our native 2x upscaler (SPAN S, distilled in experiments/exp16).
# The teacher it was distilled from (4x-UltraSharp) is only reachable from experiments and tests.
WEIGHTS = Path("models/hdfyer/hdfyer-2x-s-r2.pth")
SCALE = 2
STRIP_HEIGHT = 512   # input rows; every strip of every page is exactly this tall (see digest/strips.py)
BATCH = "auto"       # strips per forward pass: "auto" reads the GPU's profile (digest/batch_profile.py), or a number

# CPU finishing (sharpen, cleanup, encode) runs on threads while the GPU works on the next pages.
CPU_WORKERS = min(4, os.cpu_count() or 1)
MAX_INFLIGHT = 3   # stitched pages queued or being finished at once; a 2x page is ~45 MB (a teacher 4x page ~180 MB)

# Chapters at once, each in its own process with its own copy of the model (see parallel.py).
PARALLEL = 1


def process_chapter_v2(cbz_path: Path, model: torch.nn.Module, output_path: Path, scale: int = SCALE,
                       cpu_workers: int = CPU_WORKERS, max_inflight: int = MAX_INFLIGHT, overlap: bool = True,
                       strip_height: int = STRIP_HEIGHT, batch_size=1, quiet: bool = False, pages=None):
    """
    Upscales one chapter with hdfyer into output_path. batch_size is a number, or a function of strip height and width.
    quiet drops the per-chapter progress output (--parallel workers); fallbacks are still printed.
    pages: the chapter's (name, image) list if it was already loaded (see prefetched); otherwise it is loaded here.
    """
    log = (lambda *args, **kwargs: None) if quiet else print
    log(f"Loading pages from {cbz_path}...")
    loaded = reader.load_chapter(cbz_path, reader.decode_threads) if pages is None else pages
    pages = [image for _, image in loaded]
    page_names = {page_id: name for page_id, (name, _) in enumerate(loaded)}
    del loaded
    log(f"Loaded {len(pages)} pages.")

    log("Chunking pages...")
    all_chunks = []
    for page_id, image in enumerate(pages):
        page_chunks = chunk_page(image, page_id, strip_height)
        for chunk in page_chunks:
            chunk.page_name = page_names[page_id]   # so a fallback can say which page it happened on
        all_chunks.extend(page_chunks)
    log(f"Total chunks: {len(all_chunks)}")

    batches = build_batches(all_chunks, batch_size)
    log(f"Total batches: {len(batches)}")

    # Builds any missing TensorRT engine now, so the one-time cost isn't hidden inside the progress bar.
    prepare_chapter(all_chunks, scale)

    page_buffer = PageBuffer(scale)
    # A worker's exception surfaces in submit() or when the finisher closes, and fails the chapter.
    with Packager(output_path, 'jpg', 92) as packager, PageFinisher(packager, cpu_workers, max_inflight, overlap) as finisher:
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
    backend = select_backend(backend, model, weights, strip_height=STRIP_HEIGHT, round_output=True,
                             trt_batch=batch_size, arch="span")
    if engine._runtime.trt is not None:
        engine._runtime.trt.allow_build = allow_build
    if requested == "auto" and backend != "trt":
        engine.report_fallback("backend unavailable", f"TensorRT is not installed (pip install tensorrt-cu12 onnx); using {backend}")
    return model, backend, batch_size


def process_chapter(cbz_path: Path, relative: Path, output_dir: Path, model, batch_size, cpu_workers: int, max_inflight: int,
                    overlap: bool, quiet: bool = False, pages=None) -> float:
    """One chapter into output_dir/relative. Returns its time in seconds."""
    output_path = output_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to .part and rename on success, so an interrupted chapter isn't mistaken for a finished one on resume.
    chapter_start = time.perf_counter()
    part_path = output_path.with_name(output_path.name + ".part")
    process_chapter_v2(cbz_path, model, part_path, SCALE, cpu_workers, max_inflight, overlap,
                       STRIP_HEIGHT, batch_size, quiet=quiet, pages=pages)
    part_path.replace(output_path)
    torch.cuda.empty_cache()
    return time.perf_counter() - chapter_start


def process_all(manwha_dir: Path, output_dir: Path, weights: Path = WEIGHTS, backend: str = BACKEND, batch: str | int = BATCH,
                cpu_workers: int = CPU_WORKERS, max_inflight: int = MAX_INFLIGHT, overlap: bool = True, parallel: int = PARALLEL):
    """Upscales the library with hdfyer, one chapter at a time, or `parallel` chapters at once in worker processes."""
    if parallel > 1:
        from hdfyer.parallel import process_all_parallel
        return process_all_parallel(manwha_dir, output_dir, weights, backend, batch, cpu_workers, max_inflight, overlap, parallel)

    print(f"Loading hdfyer ({weights})...")
    from hdfyer.parallel import configure_threads
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
