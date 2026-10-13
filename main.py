import os
# Must be set before torch is imported: lets the allocator grow segments instead of fragmenting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import cv2
import time
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from pathlib import Path
from core import extractor
from core.model_loader import load_rrdbnet
from core.packager import Packager
from pipeline.page_buffer import PageBuffer
from profiler.cache import cache_check, save_cache, load_cache
from profiler.scanner import scan_library
from profiler.runner import run_profiler
from profiler.benchmark import build_map, get_lookup
from inference.chunker import chunk_page
from inference.batcher import build_batches
from inference.engine import BACKEND, BACKENDS, batch_upscale, engine_stats, prepare_chapter, select_backend
from postprocessing.sharpen import sharpen
from postprocessing.cleanup import cleanup_near_black

cudnn.benchmark = True

# True restores the old strips: tallest strip that fits VRAM, 128-row overlap, short last strip.
LEGACY_STRIPS = False


def run_profiler_pipeline():
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


def process_chapter_v2(cbz_path: Path, model: torch.nn.Module, benchmark_map: dict, output_path: Path, scale: int = 4, legacy_strips: bool = LEGACY_STRIPS):
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
        all_chunks.extend(chunk_page(image, page_id, benchmark_map, legacy_strips=legacy_strips))
    print(f"Total chunks: {len(all_chunks)}")

    batches = build_batches(all_chunks, benchmark_map)
    print(f"Total batches: {len(batches)}")

    # Builds any missing TensorRT engine now, so the one-time cost isn't hidden inside the progress bar.
    prepare_chapter(all_chunks, scale)

    page_buffer = PageBuffer(scale)
    with Packager(output_path, 'jpg', 92) as packager:
        for batch in tqdm(batches, desc="Upscaling batches"):
            results = batch_upscale(batch, model, scale)
            for upscaled_array, meta in results:
                completed_page = page_buffer.add(upscaled_array, meta)
                if completed_page is not None:
                    completed_page = sharpen(completed_page)
                    completed_page = cleanup_near_black(completed_page)
                    h, w = completed_page.shape[:2]
                    completed_page = cv2.resize(completed_page, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)
                    packager.add_image(page_names[meta.page_id], completed_page)

    stats = engine_stats()
    if stats["switches"]:
        print(f"TensorRT: {stats['switches']} engine switches, {sum(stats['load_seconds']):.1f}s loading engines.")
    print(f"Output written to {output_path}")


def process_all(manwha_dir: Path, output_dir: Path, model_path: Path, scale: int = 4, backend: str = BACKEND):
    print("Loading model...")
    model = load_rrdbnet(model_path)

    print("Loading benchmark map...")
    benchmark_map = load_cache()
    select_backend(backend, model, model_path, benchmark_map)

    cbz_files = sorted(manwha_dir.rglob("*.cbz"))
    print(f"Found {len(cbz_files)} chapters.")

    total_start = time.perf_counter()
    completed = 0

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
        process_chapter_v2(cbz_path, model, benchmark_map, part_path, scale, LEGACY_STRIPS)
        part_path.replace(output_path)
        torch.cuda.empty_cache()
        completed += 1
        elapsed = time.perf_counter() - chapter_start
        tqdm.write(f"Done: {relative} in {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_start
    avg = total_elapsed / completed if completed else 0
    print(f"\nFinished {completed} chapters in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min), avg {avg:.1f}s/chapter.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale every chapter in Manwhas/ into results/.")
    parser.add_argument("--backend", choices=BACKENDS, default=BACKEND,
                        help="inference runtime: auto picks TensorRT if installed, else torch.compile, else eager (default: %(default)s)")
    args = parser.parse_args()

    run_profiler_pipeline()
    process_all(Path("Manwhas"), Path("results"), Path("models/4x-UltraSharp.pth"), backend=args.backend)
