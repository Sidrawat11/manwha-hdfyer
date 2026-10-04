"""Chunk/reassemble round-trip test, no model inference.

Chunks every page of a CBZ, "upscales" each chunk with a nearest-neighbour resize,
reassembles, and checks the result matches a nearest-neighbour resize of the full
page. Also covers a small page that fits without chunking.

Usage: python test.py [path/to/chapter.cbz]   (defaults to the first CBZ in Manwhas/)
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from core import extractor
from profiler.cache import load_cache
from inference.chunker import chunk_page
from pipeline.reassembler import reassemble

SCALE = 4


def fake_upscale(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)


def round_trip(image: np.ndarray, benchmark_map: dict) -> tuple[int, int]:
    """Returns (chunk count, max pixel difference vs. the expected page)."""
    chunks = chunk_page(image, 0, benchmark_map)
    stitched = reassemble([(fake_upscale(c.page_slice), c) for c in chunks], SCALE)
    expected = fake_upscale(image)
    assert stitched.shape == expected.shape, f"shape {stitched.shape} != {expected.shape}"
    return len(chunks), int(np.abs(stitched.astype(int) - expected.astype(int)).max())


cbz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("Manwhas").rglob("*.cbz"))[0]
benchmark_map = load_cache()
failures = 0

small = np.random.default_rng(0).integers(0, 256, (400, 300, 3), dtype=np.uint8)
cases = [("synthetic 300x400", small)]
cases += [(name, image) for name, image, _ in extractor.extract_images(cbz_path)]

print(f"Testing {len(cases) - 1} pages from {cbz_path}")
for name, image in cases:
    try:
        n, diff = round_trip(image, benchmark_map)
        ok = diff <= 1  # blending identical overlaps can round down by 1
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: {image.shape[1]}x{image.shape[0]}, {n} chunks, max diff {diff}")
    except AssertionError as e:
        ok = False
        print(f"  FAIL {name}: {e}")
    failures += not ok

print(f"\n{len(cases) - failures}/{len(cases)} passed")
sys.exit(1 if failures else 0)
