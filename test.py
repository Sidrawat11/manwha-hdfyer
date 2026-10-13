"""Chunk/reassemble round-trip test, no model inference.

Chunks every page of a CBZ, "upscales" each chunk with a nearest-neighbour resize,
reassembles, and checks the result matches a nearest-neighbour resize of the full
page. Also covers a small page that fits without chunking.

Every case runs in both strip modes (smarter strips and legacy). A synthetic sweep
then covers widths 690, 800 and 1080 at every page height from 130 to 9000 in steps of 7.

Last come the inference-backend checks (engine cache key, backend selection, shape routing).
They touch no GPU either.

Usage: python test.py [path/to/chapter.cbz]   (defaults to the first CBZ in Manwhas/)
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from core import extractor
from profiler.cache import load_cache
from inference import engine, trt_backend
from inference.chunker import chunk_page, get_strip_height
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



def backend_checks(benchmark_map: dict) -> list[tuple[str, bool]]:
    """(description, passed) for each inference-backend check. None of them needs a GPU or TensorRT."""
    checks = []

    # Engine cache key: any change to shape, weights, GPU or TensorRT version must give a different file.
    base = dict(height=512, width=800, model_name="4x-UltraSharp.pth", model_hash="a" * 64,
                gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU", trt_version="11.4.0.106")
    key = trt_backend.cache_key(**base)
    checks.append(("cache key is stable", key == trt_backend.cache_key(**base)))
    checks.append(("cache key is a plain file name", "/" not in key and " " not in key and key.endswith(".plan")))
    changes = {"height": 516, "width": 720, "model_name": "other.pth", "model_hash": "b" + "a" * 63,
               "gpu_name": "NVIDIA H200", "trt_version": "11.5.0.1"}
    for field, value in changes.items():
        checks.append((f"cache key changes with {field}", trt_backend.cache_key(**{**base, field: value}) != key))
    checks.append(("cache key ignores the hash past 12 chars",
                   trt_backend.cache_key(**{**base, "model_hash": "a" * 12 + "b" * 52}) == key))

    # Backend selection: trt -> compiled -> eager, depending on what can run.
    resolve = engine.resolve_backend
    checks.append(("auto picks trt when tensorrt imports", resolve("auto", trt_ok=True, compile_ok=True) == "trt"))
    checks.append(("auto falls back to compiled without tensorrt", resolve("auto", trt_ok=False, compile_ok=True) == "compiled"))
    checks.append(("auto falls back to eager without either", resolve("auto", trt_ok=False, compile_ok=False) == "eager"))
    checks.append(("trt falls back without tensorrt", resolve("trt", trt_ok=False, compile_ok=True) == "compiled"))
    checks.append(("compiled never picks trt", resolve("compiled", trt_ok=True, compile_ok=True) == "compiled"))
    checks.append(("eager stays eager", resolve("eager", trt_ok=True, compile_ok=True) == "eager"))

    # The same with tensorrt really not importable: None in sys.modules makes `import tensorrt` raise.
    saved = sys.modules.get("tensorrt", ...)
    sys.modules["tensorrt"] = None
    try:
        checks.append(("is_available is False when tensorrt can't be imported", trt_backend.is_available() is False))
        checks.append(("auto avoids trt when tensorrt can't be imported", resolve("auto") != "trt"))
        checks.append(("trt falls back when tensorrt can't be imported", resolve("trt") != "trt"))
    finally:
        if saved is ...:
            del sys.modules["tensorrt"]
        else:
            sys.modules["tensorrt"] = saved

    # Routing: only a width's standard strip shape goes to the backend, everything else is eager.
    saved_runtime = engine._runtime
    try:
        for backend in ("trt", "compiled"):
            runtime = engine._Runtime()
            runtime.backend, runtime.benchmark_map, runtime.failed_shapes = backend, benchmark_map, set()
            engine._runtime = runtime
            for w in SWEEP_WIDTHS:
                strip = get_strip_height(w, benchmark_map)
                checks.append((f"{backend}: standard {w}x{strip} strip uses {backend}", engine.route(strip, w) == backend))
                others = [strip - 1, strip + 1, strip + 16, 130, 400]
                checks.append((f"{backend}: non-standard heights at width {w} run eager",
                               all(engine.route(h, w) == "eager" for h in others if h != strip)))
            strip = get_strip_height(800, benchmark_map)
            runtime.failed_shapes.add((strip, 800))
            checks.append((f"{backend}: a shape that failed runs eager", engine.route(strip, 800) == "eager"))
            other = get_strip_height(690, benchmark_map)
            checks.append((f"{backend}: other shapes keep {backend} after a failure", engine.route(other, 690) == backend))

        runtime = engine._Runtime()
        runtime.backend, runtime.benchmark_map, runtime.failed_shapes = "eager", benchmark_map, set()
        engine._runtime = runtime
        checks.append(("eager: standard strips run eager", engine.route(get_strip_height(800, benchmark_map), 800) == "eager"))

        # Real pages, real planner: every chunk routed to trt must be the standard strip of its width.
        runtime.backend = "trt"
        routed = {"trt": 0, "eager": 0}
        only_standard = True
        for w, h in [(800, 400), (800, 898), (800, 1000), (800, 4540), (690, 3000), (1080, 9000)]:
            for chunk in chunk_page(np.broadcast_to(np.zeros((h, 1, 3), np.uint8), (h, w, 3)), 0, benchmark_map):
                ch, cw = chunk.page_slice.shape[:2]
                routed[engine.route(ch, cw)] += 1
                if engine.route(ch, cw) == "trt" and ch != get_strip_height(cw, benchmark_map):
                    only_standard = False
        checks.append(("planned pages: only standard strips reach trt", only_standard and routed["trt"] > 0))
        checks.append(("planned pages: single-chunk and grown strips run eager", routed["eager"] >= 4))
    finally:
        engine._runtime = saved_runtime

    checks.append(("padded shape rounds up to a multiple of 4",
                   engine._padded_shape(593, 690) == (596, 692) and engine._padded_shape(512, 800) == (512, 800)))
    return checks


print("Backend checks (no GPU)")
checks = backend_checks(benchmark_map)
for description, ok in checks:
    if not ok:
        print(f"  FAIL {description}")
backend_failures = sum(not ok for _, ok in checks)
print(f"  {'OK  ' if not backend_failures else 'FAIL'} {len(checks) - backend_failures}/{len(checks)} backend checks")

print(f"\n{total - failures}/{total} passed")
print(f"{len(checks) - backend_failures}/{len(checks)} backend checks passed")
sys.exit(1 if failures or backend_failures else 0)
