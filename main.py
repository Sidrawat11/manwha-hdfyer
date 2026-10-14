import os
# Must be set before torch is imported: lets the allocator grow segments instead of fragmenting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import time
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from pathlib import Path
from core import extractor
from core.model_loader import load_student
from core.packager import Packager
from pipeline.page_buffer import PageBuffer
from pipeline.finisher import PageFinisher
from inference.chunker import chunk_page
from inference import engine, trt_backend
from inference.batcher import build_batches
from inference.engine import BACKEND, BACKENDS, batch_upscale, engine_stats, prepare_chapter, select_backend
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
                       strip_height: int | None = None, batch_size=None):
    """
    Upscales one chapter with whatever model the caller loaded. `scale` is that model's output scale (4 or 2);
    the CBZ is 2x either way. hdfyer passes strip_height and batch_size (a number, or a function of strip height
    and width) and no benchmark map; the defaults are the teacher's, for experiments.
    """
    print(f"Loading pages from {cbz_path}...")
    pages = []
    page_names = {}
    for page_id, (name, image, _) in enumerate(extractor.extract_images(cbz_path)):
        pages.append(image)
        page_names[page_id] = name
    print(f"Loaded {len(pages)} pages.")

    print("Chunking pages...")
    all_chunks = []
    for page_id, image in enumerate(pages):
        page_chunks = chunk_page(image, page_id, benchmark_map, legacy_strips=legacy_strips, strip_height=strip_height)
        for chunk in page_chunks:
            chunk.page_name = page_names[page_id]   # so a fallback can say which page it happened on
        all_chunks.extend(page_chunks)
    print(f"Total chunks: {len(all_chunks)}")

    batches = build_batches(all_chunks, benchmark_map, batch_size)
    print(f"Total batches: {len(batches)}")

    # Builds any missing TensorRT engine now, so the one-time cost isn't hidden inside the progress bar.
    prepare_chapter(all_chunks, scale)

    page_buffer = PageBuffer(scale)
    # A worker's exception surfaces in submit() or when the finisher closes, and fails the chapter.
    with Packager(output_path, 'jpg', 92) as packager, PageFinisher(packager, cpu_workers, max_inflight, overlap, scale) as finisher:
        for batch in tqdm(batches, desc="Upscaling batches"):
            results = batch_upscale(batch, model, scale)
            for upscaled_array, meta in results:
                completed_page = page_buffer.add(upscaled_array, meta)
                if completed_page is not None:
                    finisher.submit(page_names[meta.page_id], completed_page)
                    del completed_page

    stats = engine_stats()
    if stats["switches"]:
        print(f"TensorRT: {stats['switches']} engine switches, {sum(stats['load_seconds']):.1f}s loading engines.")
    print(f"Output written to {output_path}")


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


def print_run_summary(backend: str) -> None:
    """End-of-run report: engines built, and every fallback from the selected backend."""
    builds = engine_stats()["build_seconds"]
    if builds:
        print(f"TensorRT engines built this run: {len(builds)}, {sum(builds):.0f}s in total.")
    events = engine.fallback_events()
    if not events:
        print(f"Fallbacks: none. Every strip ran on {backend}.")
        return
    print(f"Fallbacks: {len(events)}")
    for event in events:
        print("  " + engine.format_event(event))


def process_all(manwha_dir: Path, output_dir: Path, weights: Path = WEIGHTS, backend: str = BACKEND, batch: str | int = BATCH,
                cpu_workers: int = CPU_WORKERS, max_inflight: int = MAX_INFLIGHT, overlap: bool = True):
    """Upscales the library with hdfyer."""
    print(f"Loading hdfyer ({weights})...")
    model = load_student(weights)

    # auto means TensorRT, or eager if TensorRT isn't installed. Nothing here can reach another model.
    requested = backend
    if backend == "auto":
        backend = "trt" if trt_backend.is_available() else "eager"
    batch_size = batch
    if batch == "auto":
        weights_id = f"{weights.name}-{trt_backend.file_sha256(weights)[:trt_backend.HASH_LENGTH]}"
        batch_size = BatchProfile(model, weights_id, STRIP_HEIGHT, before_measuring=engine.free_gpu)
    backend = select_backend(backend, model, weights, None, strip_height=STRIP_HEIGHT, round_output=True,
                             trt_batch=batch_size, arch="span")
    if requested == "auto" and backend != "trt":
        engine.report_fallback("backend unavailable", f"TensorRT is not installed (pip install tensorrt-cu12 onnx); using {backend}")

    cbz_files = sorted(manwha_dir.rglob("*.cbz"))
    print(f"Found {len(cbz_files)} chapters.")

    total_start = time.perf_counter()
    completed = 0

    try:
        for cbz_path in tqdm(cbz_files, desc="Processing chapters"):
            relative = cbz_path.relative_to(manwha_dir)
            output_path = output_dir / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                tqdm.write(f"Skipping, already processed: {relative}")
                continue

            # Write to .part and rename on success, so an interrupted chapter isn't mistaken for a finished one on resume.
            chapter_start = time.perf_counter()
            part_path = output_path.with_name(output_path.name + ".part")
            engine.set_context(str(relative))
            process_chapter_v2(cbz_path, model, None, part_path, SCALE, False, cpu_workers, max_inflight, overlap,
                               STRIP_HEIGHT, batch_size)
            part_path.replace(output_path)
            torch.cuda.empty_cache()
            completed += 1
            elapsed = time.perf_counter() - chapter_start
            tqdm.write(f"Done: {relative} in {elapsed:.1f}s")
    finally:
        # Also after a failure or Ctrl+C, so fallbacks that led up to it are not lost.
        total_elapsed = time.perf_counter() - total_start
        avg = total_elapsed / completed if completed else 0
        print(f"\nFinished {completed} chapters in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min), avg {avg:.1f}s/chapter.")
        print_run_summary(backend)


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
    args = parser.parse_args()
    if args.cpu_workers < 1 or args.max_inflight < 1:
        parser.error("--cpu-workers and --max-inflight must be at least 1")
    if not args.weights.exists():
        parser.error(f"hdfyer weights not found: {args.weights}")

    process_all(Path("Manwhas"), Path("results"), args.weights, backend=args.backend, batch=args.batch,
                cpu_workers=args.cpu_workers, max_inflight=args.max_inflight, overlap=not args.no_overlap)
