"""Chunk/reassemble round-trip test, no model inference.

Chunks every page of a CBZ, "upscales" each chunk with a nearest-neighbour resize,
reassembles, and checks the result matches a nearest-neighbour resize of the full
page. Also covers a small page that fits without chunking.

Every case runs in both strip modes (smarter strips and legacy). A synthetic sweep
then covers widths 690, 800 and 1080 at every page height from 130 to 9000 in steps of 7,
and repeats it for the 2x student's strips (fixed 512 rows, scale 2).

Last come the inference-backend checks (engine cache key, backend selection, shape routing).
They touch no GPU either. Finally the page-finishing checks run fake 4x pages through the
overlapped and the serial path and compare the two CBZs.

Usage: python test.py [path/to/chapter.cbz]   (defaults to the first CBZ in Manwhas/)
"""

import sys
import tempfile
import threading
import time
import zipfile
import cv2
import numpy as np
from pathlib import Path
import torch
from core import extractor
from core.model_loader import fold_reparam
from profiler.cache import load_cache
from inference import engine, trt_backend
from inference.chunker import chunk_page, get_strip_height
from inference.batcher import build_batches
from pipeline.reassembler import reassemble
from pipeline import finisher as finisher_module
from pipeline.finisher import PageFinisher
from core.packager import Packager
from postprocessing.sharpen import sharpen
from postprocessing.cleanup import cleanup_near_black

SCALE = 4
MODES = [(False, "smarter"), (True, "legacy")]
SWEEP_WIDTHS = [690, 800, 1080]
SWEEP_HEIGHTS = range(130, 9001, 7)
SWEEP_COLUMNS = 8
STUDENT_SCALE = 2
STUDENT_STRIP_HEIGHT = 512


def fake_upscale(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def round_trip(image: np.ndarray, benchmark_map: dict | None, legacy_strips: bool, upscale=fake_upscale,
               scale: int = SCALE, strip_height: int | None = None) -> tuple[int, int]:
    """Returns (chunk count, max pixel difference vs. the expected page)."""
    chunks = chunk_page(image, 0, benchmark_map, legacy_strips=legacy_strips, strip_height=strip_height)
    if not legacy_strips:
        heights = {c.y_end - c.y_start for c in chunks}
        assert len(heights) == 1, f"smarter strips must share one height, got {sorted(heights)}"
    stitched = reassemble([(upscale(c.page_slice), c) for c in chunks], scale)
    expected = upscale(image)
    assert stitched.shape[0] == image.shape[0] * scale, f"{stitched.shape[0]} rows != {image.shape[0] * scale}"
    assert stitched.shape == expected.shape, f"shape {stitched.shape} != {expected.shape}"
    return len(chunks), int(np.abs(stitched.astype(int) - expected.astype(int)).max())


def fake_upscale_column(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """fake_upscale on the first few columns only. Sweep pages are constant along x, so nothing is lost."""
    return fake_upscale(np.ascontiguousarray(image[:, :SWEEP_COLUMNS]), scale)


def fake_upscale_column_2x(image: np.ndarray) -> np.ndarray:
    return fake_upscale_column(image, STUDENT_SCALE)


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
# (label, legacy_strips, stand-in upscaler, scale, fixed strip height, benchmark map)
SWEEPS = [(mode, legacy, fake_upscale_column, SCALE, None, benchmark_map) for legacy, mode in MODES]
SWEEPS.append(("student", False, fake_upscale_column_2x, STUDENT_SCALE, STUDENT_STRIP_HEIGHT, None))
for mode, legacy_strips, upscale, scale, strip_height, sweep_map in SWEEPS:
    for w in SWEEP_WIDTHS:
        failed, worst, most_chunks = 0, 0, 0
        for h in SWEEP_HEIGHTS:
            try:
                n, diff = round_trip(sweep_page(rng, w, h), sweep_map, legacy_strips, upscale, scale, strip_height)
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

    # Student: one fixed strip height for every width, no benchmark map.
    saved_runtime = engine._runtime
    try:
        runtime = engine._Runtime()
        runtime.backend, runtime.benchmark_map, runtime.failed_shapes = "trt", None, set()
        runtime.strip_height = STUDENT_STRIP_HEIGHT
        engine._runtime = runtime
        checks.append(("student: 512-row strips use the backend at any width",
                       all(engine.route(STUDENT_STRIP_HEIGHT, w) == "trt" for w in (436, 690, 800, 1080, 3200))))
        checks.append(("student: other heights run eager",
                       all(engine.route(h, 800) == "eager" for h in (130, 379, 511, 513, 528, 898))))
        shapes = set()
        for w, h in [(800, 300), (800, 512), (800, 1000), (800, 4540), (690, 3000), (1080, 9000)]:
            page = np.broadcast_to(np.zeros((h, 1, 3), np.uint8), (h, w, 3))
            shapes |= {c.page_slice.shape[:2] for c in chunk_page(page, 0, None, strip_height=STUDENT_STRIP_HEIGHT)}
        checks.append(("student: planned strips are 512 rows except short and grown two-strip pages",
                       shapes == {(300, 800), (512, 800), (516, 800), (512, 690), (512, 1080)}))
    finally:
        engine._runtime = saved_runtime

    # Batching: a fixed batch size groups same-shape strips across pages and never mixes shapes.
    chunks = []
    for page_id, (w, h) in enumerate([(800, 4540), (800, 3000), (690, 2000), (800, 300), (800, 1000)]):
        page = np.broadcast_to(np.zeros((h, 1, 3), np.uint8), (h, w, 3))
        chunks += chunk_page(page, page_id, None, strip_height=STUDENT_STRIP_HEIGHT)
    batches = build_batches(chunks, None, batch_size=8)
    checks.append(("batching: every chunk is in exactly one batch",
                   sorted(id(c) for b in batches for c in b) == sorted(id(c) for c in chunks)))
    checks.append(("batching: a batch holds one shape and at most 8 strips",
                   all(len(b) <= 8 and len({c.page_slice.shape for c in b}) == 1 for b in batches)))
    checks.append(("batching: only the last batch of a shape is short",
                   [len(b) for b in batches if b[0].page_slice.shape[:2] == (512, 800)] == [8, 8, 1]))
    checks.append(("batching: strips of different pages share a batch",
                   any(len({c.page_id for c in b}) > 1 for b in batches)))
    checks.append(("batching: batch size 1 gives one strip per batch",
                   all(len(b) == 1 for b in build_batches(chunks, None, batch_size=1))))

    # SPAN's convolutions must be folded once at load, not on every forward pass.
    class FakeConv3XC:
        def __init__(self):
            self.folds, self.eval_conv = 0, object()
        def update_params(self):
            self.folds += 1
    class FakeModel:
        def __init__(self):
            self.convs = [FakeConv3XC(), FakeConv3XC()]
        def modules(self):
            return [self, *self.convs]
    fake = FakeModel()
    folded = fold_reparam(fake)
    for conv in fake.convs:
        conv.update_params()   # what Conv3XC.forward does on every pass
    checks.append(("fold_reparam folds each conv once and stops later refolds",
                   folded == 2 and [c.folds for c in fake.convs] == [1, 1]))

    scaled = torch.tensor([[[[0.4 / 255, 0.6 / 255, 254.5 / 255, 2.0]]]]).repeat(1, 3, 1, 1)
    checks.append(("postprocess truncates by default and rounds for the student",
                   engine._postprocess(scaled)[0, :, 0].tolist() == [0, 0, 254, 255]
                   and engine._postprocess(scaled, round_output=True)[0, :, 0].tolist() == [0, 1, 254, 255]))

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



def write_cbz(path: Path, pages: list[tuple[str, np.ndarray]], overlap: bool, workers: int = 4, max_inflight: int = 3, scale: int = SCALE) -> int:
    """Runs fake stitched pages through the finishing path into a CBZ. Returns the most pages that were in flight at once."""
    most_inflight = 0
    with Packager(path, 'jpg', 92) as packager, PageFinisher(packager, workers, max_inflight, overlap, scale) as page_finisher:
        for name, page in pages:
            page_finisher.submit(name, page)
            most_inflight = max(most_inflight, len(page_finisher._pending))
    return most_inflight


def cbz_entries(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as z:
        return [(name, z.read(name)) for name in z.namelist()]


def finisher_checks() -> list[tuple[str, bool]]:
    """(description, passed) for each page-finishing check. No GPU: the pages are random arrays."""
    checks = []
    rng = np.random.default_rng(2)
    # Big pages first, so with several workers the later pages finish before the earlier ones.
    heights = [2400, 1800, 96, 64, 1200, 32, 900, 48, 600, 16, 300, 8]
    pages = [(f"{i:03d}__page.png", rng.integers(0, 256, (h, 640, 3), dtype=np.uint8)) for i, h in enumerate(heights)]
    pages[3][1][:] = 5   # a near-black page, for the cleanup step
    expected_names = [str(Path(name).with_suffix(".jpg")) for name, _ in pages]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_cbz(tmp / "serial.cbz", pages, overlap=False)
        serial = cbz_entries(tmp / "serial.cbz")
        checks.append(("serial: every page written, in order", [name for name, _ in serial] == expected_names))
        decoded = cv2.imdecode(np.frombuffer(serial[0][1], np.uint8), cv2.IMREAD_COLOR)
        checks.append(("serial: pages are halved JPEGs", decoded is not None and decoded.shape == (1200, 320, 3)))
        black = cv2.imdecode(np.frombuffer(serial[3][1], np.uint8), cv2.IMREAD_COLOR)
        checks.append(("serial: near-black page is cleaned to black", black is not None and int(black.max()) == 0))

        for workers, max_inflight in [(4, 3), (1, 1), (8, 2), (2, 6)]:
            label = f"overlap, {workers} workers, {max_inflight} in flight"
            most = write_cbz(tmp / "overlap.cbz", pages, overlap=True, workers=workers, max_inflight=max_inflight)
            overlapped = cbz_entries(tmp / "overlap.cbz")
            checks.append((f"{label}: same page order as serial", [n for n, _ in overlapped] == [n for n, _ in serial]))
            checks.append((f"{label}: every page byte-identical to serial", overlapped == serial))
            checks.append((f"{label}: never more than {max_inflight} pages in flight", 1 <= most <= max_inflight))

        # 2x pages (native 2x model): kept at their size, sharpened with radius 0.5, cleaned, same in both paths.
        write_cbz(tmp / "serial2x.cbz", pages, overlap=False, scale=STUDENT_SCALE)
        write_cbz(tmp / "overlap2x.cbz", pages, overlap=True, scale=STUDENT_SCALE)
        serial2x = cbz_entries(tmp / "serial2x.cbz")
        checks.append(("2x pages: overlapped CBZ byte-identical to serial",
                       (tmp / "serial2x.cbz").read_bytes() == (tmp / "overlap2x.cbz").read_bytes()))
        decoded = cv2.imdecode(np.frombuffer(serial2x[0][1], np.uint8), cv2.IMREAD_COLOR)
        checks.append(("2x pages: not halved", decoded is not None and decoded.shape == (2400, 640, 3)))
        expected = cleanup_near_black(sharpen(pages[4][1], radius=0.5))
        ok, encoded = cv2.imencode(".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, 92])
        checks.append(("2x pages: sharpen radius 0.5 then black cleanup, nothing else", ok and encoded.tobytes() == serial2x[4][1]))
        black = cv2.imdecode(np.frombuffer(serial2x[3][1], np.uint8), cv2.IMREAD_COLOR)
        checks.append(("2x pages: near-black page is cleaned to black", black is not None and int(black.max()) == 0))
        try:
            write_cbz(tmp / "bad.cbz", pages[:1], overlap=False, scale=3)
            rejected = False
        except ValueError:
            rejected = True
        checks.append(("a scale other than 2 or 4 is rejected", rejected))

        # Pages really are finished off the calling thread, and several at once.
        real_finish = finisher_module.finish_page
        threads, running, most_running, lock = set(), 0, 0, threading.Lock()

        def tracked_finish(page, packager, scale=SCALE):
            nonlocal running, most_running
            with lock:
                threads.add(threading.current_thread().name)
                running += 1
                most_running = max(most_running, running)
            time.sleep(0.05)
            try:
                return real_finish(page, packager, scale)
            finally:
                with lock:
                    running -= 1

        def failing_finish(page, packager, scale=SCALE):
            if page.shape[0] == 96:
                raise RuntimeError("simulated finishing failure")
            return real_finish(page, packager, scale)

        try:
            finisher_module.finish_page = tracked_finish
            write_cbz(tmp / "tracked.cbz", pages, overlap=True, workers=4, max_inflight=3)
            checks.append(("overlap: pages are finished on worker threads",
                           threads and threading.current_thread().name not in threads))
            checks.append(("overlap: several pages are finished at once, within the limit", 2 <= most_running <= 3))
            threads.clear()
            write_cbz(tmp / "tracked.cbz", pages, overlap=False)
            checks.append(("serial: pages are finished on the calling thread", threads == {threading.current_thread().name}))

            # Same pages packaged twice, more than a zip timestamp tick (2 s) apart: the files must be byte-identical.
            finisher_module.finish_page = real_finish
            write_cbz(tmp / "first.cbz", pages, overlap=True)
            time.sleep(2.1)
            write_cbz(tmp / "second.cbz", pages, overlap=True)
            checks.append(("packaging the same pages twice gives byte-identical CBZ files",
                           (tmp / "first.cbz").read_bytes() == (tmp / "second.cbz").read_bytes()))
            checks.append(("serial and overlapped CBZ files are byte-identical",
                           (tmp / "serial.cbz").read_bytes() == (tmp / "second.cbz").read_bytes()))
            with zipfile.ZipFile(tmp / "first.cbz") as z:
                checks.append(("every zip entry has the fixed 1980-01-01 timestamp",
                               all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in z.infolist())))
                checks.append(("zip entries keep their compression and permissions",
                               all(info.compress_type == zipfile.ZIP_STORED and info.external_attr == 0o600 << 16
                                   for info in z.infolist())))

            # A failing page must fail the whole run, in both paths, and nothing after it may be written.
            finisher_module.finish_page = failing_finish
            for overlap in (True, False):
                label = "overlap" if overlap else "serial"
                try:
                    write_cbz(tmp / "failed.cbz", pages, overlap=overlap)
                    raised = False
                except RuntimeError as e:
                    raised = "simulated finishing failure" in str(e)
                checks.append((f"{label}: a worker exception fails the run", raised))
                written = [name for name, _ in cbz_entries(tmp / "failed.cbz")]
                checks.append((f"{label}: pages after the failed one are not written", written == expected_names[:2]))
        finally:
            finisher_module.finish_page = real_finish

    return checks


print("Page-finishing checks (no GPU)")
finish_checks = finisher_checks()
for description, ok in finish_checks:
    if not ok:
        print(f"  FAIL {description}")
finisher_failures = sum(not ok for _, ok in finish_checks)
print(f"  {'OK  ' if not finisher_failures else 'FAIL'} {len(finish_checks) - finisher_failures}/{len(finish_checks)} page-finishing checks")

print(f"\n{total - failures}/{total} passed")
print(f"{len(checks) - backend_failures}/{len(checks)} backend checks passed")
print(f"{len(finish_checks) - finisher_failures}/{len(finish_checks)} page-finishing checks passed")
sys.exit(1 if failures or backend_failures or finisher_failures else 0)
