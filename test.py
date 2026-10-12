"""Chunk/reassemble round-trip test, no model inference.

Chunks every page of a CBZ, "upscales" each chunk with a nearest-neighbour resize,
reassembles, and checks the result matches a nearest-neighbour resize of the full
page. Also covers a small page that fits without chunking.

Every case runs in both strip modes (smarter strips and legacy). A synthetic sweep
then covers widths 690, 800 and 1080 at every page height from 130 to 9000 in steps of 7.

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
MODES = [(False, "smarter"), (True, "legacy")]
SWEEP_WIDTHS = [690, 800, 1080]
SWEEP_HEIGHTS = range(130, 9001, 7)
SWEEP_COLUMNS = 8


def fake_upscale(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)


def round_trip(image: np.ndarray, benchmark_map: dict, legacy_strips: bool, upscale=fake_upscale) -> tuple[int, int]:
    """Returns (chunk count, max pixel difference vs. the expected page)."""
    chunks = chunk_page(image, 0, benchmark_map, legacy_strips=legacy_strips)
    if not legacy_strips:
        heights = {c.y_end - c.y_start for c in chunks}
        assert len(heights) == 1, f"smarter strips must share one height, got {sorted(heights)}"
    stitched = reassemble([(upscale(c.page_slice), c) for c in chunks], SCALE)
    expected = upscale(image)
    assert stitched.shape[0] == image.shape[0] * SCALE, f"{stitched.shape[0]} rows != {image.shape[0] * SCALE}"
    assert stitched.shape == expected.shape, f"shape {stitched.shape} != {expected.shape}"
    return len(chunks), int(np.abs(stitched.astype(int) - expected.astype(int)).max())


def fake_upscale_column(image: np.ndarray) -> np.ndarray:
    """fake_upscale on the first few columns only. Sweep pages are constant along x, so nothing is lost."""
    return fake_upscale(np.ascontiguousarray(image[:, :SWEEP_COLUMNS]))


def sweep_page(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """A w x h page of random rows, constant along x: chunking and stitching only act on rows, and this keeps the sweep fast."""
    rows = rng.integers(0, 256, (h, 1, 3), dtype=np.uint8)
    return np.broadcast_to(rows, (h, w, 3))


cbz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("Manwhas").rglob("*.cbz"))[0]
benchmark_map = load_cache()
failures = 0
total = 0

small = np.random.default_rng(0).integers(0, 256, (400, 300, 3), dtype=np.uint8)
cases = [("synthetic 300x400", small)]
cases += [(name, image) for name, image, _ in extractor.extract_images(cbz_path)]

print(f"Testing {len(cases) - 1} pages from {cbz_path}")
for legacy_strips, mode in MODES:
    print(f" {mode} strips")
    for name, image in cases:
        try:
            n, diff = round_trip(image, benchmark_map, legacy_strips)
            ok = diff <= 1  # blending identical overlaps can round down by 1
            print(f"  {'OK  ' if ok else 'FAIL'} {name}: {image.shape[1]}x{image.shape[0]}, {n} chunks, max diff {diff}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL {name}: {e}")
        total += 1
        failures += not ok

print(f"Sweep: widths {SWEEP_WIDTHS}, heights {SWEEP_HEIGHTS.start}-{SWEEP_HEIGHTS[-1]} in steps of {SWEEP_HEIGHTS.step}")
rng = np.random.default_rng(1)
for legacy_strips, mode in MODES:
    for w in SWEEP_WIDTHS:
        failed, worst, most_chunks = 0, 0, 0
        for h in SWEEP_HEIGHTS:
            try:
                n, diff = round_trip(sweep_page(rng, w, h), benchmark_map, legacy_strips, fake_upscale_column)
                ok = diff <= 1
                worst, most_chunks = max(worst, diff), max(most_chunks, n)
                if not ok:
                    print(f"  FAIL {mode} {w}x{h}: {n} chunks, max diff {diff}")
            except AssertionError as e:
                ok = False
                print(f"  FAIL {mode} {w}x{h}: {e}")
            failed += not ok
        total += len(SWEEP_HEIGHTS)
        failures += failed
        print(f"  {'OK  ' if not failed else 'FAIL'} {mode} strips, width {w}: {len(SWEEP_HEIGHTS) - failed}/{len(SWEEP_HEIGHTS)} heights, "
              f"up to {most_chunks} chunks, max diff {worst}")

print(f"\n{total - failures}/{total} passed")
sys.exit(1 if failures else 0)
