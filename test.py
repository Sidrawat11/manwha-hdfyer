"""Chunk/reassemble round-trip test, no model inference.

Chunks every page of a CBZ, "upscales" each chunk with a nearest-neighbour resize,
reassembles, and checks the result matches a nearest-neighbour resize of the full
page. Also covers a small page that fits without chunking.

Every case runs in both strip modes (smarter strips and legacy). A synthetic sweep
then covers widths 690, 800 and 1080 at every page height from 130 to 9000 in steps of 7,
and repeats it for hdfyer's fixed-shape strips (512 rows, anchored or padded, scale 2).

Last come the inference-backend checks (engine cache key, backend selection, shape routing).
They touch no GPU either. Finally the page-finishing checks run fake 4x pages through the
overlapped and the serial path and compare the two CBZs.

Usage: python test.py [path/to/chapter.cbz]   (defaults to the first CBZ in Manwhas/)
"""

import json
import os
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
from inference.chunker import ChunkMeta, chunk_page, crop_upscaled, get_strip_height
from inference.batcher import build_batches
from profiler import hdfyer as hdfyer_profile
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
SHORT_HEIGHTS = [1, 2, 3, 31, 32, 33, 64, 127, 255, 256, 257, 400, 510, 511, 512, 513, 543, 544, 545, 991, 992, 993, 1023, 1024, 1025]
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
        heights = {c.page_slice.shape[0] for c in chunks}
        assert len(heights) == 1, f"smarter strips must share one height, got {sorted(heights)}"
    if strip_height is not None:
        assert heights == {strip_height}, f"fixed-shape strips must all be {strip_height} rows, got {sorted(heights)}"
    stitched = reassemble([(crop_upscaled(upscale(c.page_slice), c, scale), c) for c in chunks], scale)
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
SWEEPS.append(("hdfyer", False, fake_upscale_column_2x, STUDENT_SCALE, STUDENT_STRIP_HEIGHT, None))
for mode, legacy_strips, upscale, scale, strip_height, sweep_map in SWEEPS:
    # hdfyer also gets the short and boundary heights: padded pages, one strip, the anchored last strip.
    heights = list(SWEEP_HEIGHTS) + (SHORT_HEIGHTS if strip_height is not None else [])
    for w in SWEEP_WIDTHS:
        failed, worst, most_chunks = 0, 0, 0
        for h in heights:
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
        total += len(heights)
        failures += failed
        print(f"  {'OK  ' if not failed else 'FAIL'} {mode} strips, width {w}: {len(heights) - failed}/{len(heights)} heights, "
              f"up to {most_chunks} chunks, max diff {worst}")


class FakeOutOfMemory:
    """A 2x nearest-neighbour stand-in for the model that raises out-of-memory the way the GPU does."""

    def __init__(self, largest_batch: int = 99, message: str = "CUDA out of memory. Tried to allocate 4.39 GiB", fail_first: int = 0):
        self.largest_batch, self.message, self.fail_first = largest_batch, message, fail_first
        self.calls = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(x.shape[0])
        if x.shape[0] > self.largest_batch or len(self.calls) <= self.fail_first:
            raise RuntimeError(self.message)
        return torch.nn.functional.interpolate(x, scale_factor=STUDENT_SCALE, mode="nearest")


class FakeTrt:
    """Stands in for trt_backend.TrtBackend: engines of a fixed batch that can fail to build or run."""

    def __init__(self, run_model, largest_build: int = 99, build_message: str = "CUDA driver error: out of memory"):
        self.run_model, self.largest_build, self.build_message = run_model, largest_build, build_message
        self.built, self.released = [], 0
        self.switches, self.load_seconds, self.build_seconds = 0, [], []

    def has_engine(self, height, width, batch=1):
        return (height, width, batch) in self.built

    def build(self, height, width, batch=1):
        if batch > self.largest_build:
            raise trt_backend.EngineBuildError(f"engine build for {height}x{width} (batch {batch}) failed ({self.build_message})")
        self.built.append((height, width, batch))

    def release(self):
        self.released += 1

    def run(self, x):
        batch, _, height, width = x.shape
        if not self.has_engine(height, width, batch):
            self.build(height, width, batch)
        return self.run_model(x)


def profile_checks() -> list[tuple[str, bool]]:
    """hdfyer's batch profile: which batch is chosen, and that it is measured once and then read from the cache."""
    checks = []
    pick = hdfyer_profile.pick_best_batch
    checks.append(("profile: equal time per strip picks the smallest batch", pick({1: 17.6, 2: 17.7, 4: 17.5, 8: 17.8}) == 1))
    checks.append(("profile: a clearly faster batch is picked", pick({1: 20.0, 2: 15.0, 4: 12.0, 8: 11.9}) == 4))
    checks.append(("profile: a gain under 3% does not justify a bigger batch", pick({1: 10.0, 2: 9.8}) == 1 and pick({1: 10.0, 2: 9.6}) == 2))
    checks.append(("profile: a single measurement is its own best", pick({1: 58.0}) == 1))

    saved = (hdfyer_profile.cache.load_section, hdfyer_profile.cache.save_section, hdfyer_profile.profile_width)
    stored, measured, freed = {}, [], []

    def fake_profile_width(model, width, strip_height, vram_mb=None):
        measured.append(width)
        return {"best_batch": 4 if width >= 1000 else 2, "batches": {"1": {"ms_per_strip": 9.0, "peak_vram_mb": 100}}, "stopped": "test"}

    hdfyer_profile.cache.load_section = lambda key: dict(stored.get(key, {}))
    hdfyer_profile.cache.save_section = lambda key, section: stored.__setitem__(key, json.loads(json.dumps(section)))
    hdfyer_profile.profile_width = fake_profile_width
    try:
        profile = hdfyer_profile.BatchProfile(None, "weights-a", 512, before_measuring=lambda: freed.append(1))
        first = [profile(512, 800), profile(512, 800), profile(512, 1080), profile(512, 691), profile(512, 692)]
        checks.append(("profile: a width is measured once, then answered from the profile",
                       first == [2, 2, 4, 2, 2] and measured == [800, 1080, 692]))
        checks.append(("profile: the GPU is freed before each measurement", len(freed) == 3))
        checks.append(("profile: results are saved under the GPU's hdfyer section",
                       sorted(stored["hdfyer"]["widths"]) == ["1080", "692", "800"] and stored["hdfyer"]["weights"] == "weights-a"))

        measured.clear()
        again = hdfyer_profile.BatchProfile(None, "weights-a", 512)
        checks.append(("profile: a later run reads the saved profile without measuring",
                       [again(512, 800), again(512, 1080)] == [2, 4] and measured == []))
        other = hdfyer_profile.BatchProfile(None, "weights-b", 512)
        checks.append(("profile: other weights do not reuse the profile", other(512, 800) == 2 and measured == [800]))
    finally:
        hdfyer_profile.cache.load_section, hdfyer_profile.cache.save_section, hdfyer_profile.profile_width = saved
    return checks


def parallel_checks() -> list[tuple[str, bool]]:
    """--parallel plumbing that needs no GPU: the warm-up sees the same pages as the workers, resume, state hand-over."""
    import main
    from core import extractor
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cbz = tmp / "lib" / "series" / "ch1.cbz"
        cbz.parent.mkdir(parents=True)
        rng = np.random.default_rng(4)
        with zipfile.ZipFile(cbz, "w") as z:
            for name, (h, w) in [("002.jpg", (300, 690)), ("001.png", (40, 1080)), ("003.webp", (90, 800))]:
                ok, data = cv2.imencode(Path(name).suffix, rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
                z.writestr(name, data.tobytes())
            z.writestr("__MACOSX/._001.png", b"junk")
            z.writestr("ComicInfo.xml", b"<x/>")
        decoded = [(image.shape[1], image.shape[0]) for _, image, _ in extractor.extract_images(cbz)]
        checks.append(("parallel: the warm-up's header scan sees exactly the pages the workers decode",
                       extractor.page_sizes(cbz) == decoded == [(1080, 40), (690, 300), (800, 90)]))

        (tmp / "lib" / "series" / "ch2.cbz").write_bytes(cbz.read_bytes())
        (tmp / "lib" / "series" / "ch3.cbz").write_bytes(cbz.read_bytes())
        out = tmp / "out" / "series"
        out.mkdir(parents=True)
        (out / "ch1.cbz").write_bytes(b"finished")
        (out / "ch2.cbz.part").write_bytes(b"interrupted")
        todo = [str(relative) for _, relative in main.todo_chapters(tmp / "lib", tmp / "out")]
        checks.append(("parallel: finished chapters are skipped and an interrupted .part is redone",
                       todo == ["series/ch2.cbz", "series/ch3.cbz"]))

        decoded_in_threads = extractor.load_chapter(cbz, threads=3)
        checks.append(("decode ahead: threaded decoding gives the same pages, in the same order",
                       [n for n, _ in decoded_in_threads] == [n for n, _, _ in extractor.extract_images(cbz)]
                       and all(np.array_equal(a, b) for (_, a), (_, _, b) in
                               zip(decoded_in_threads, [(n, i, i) for n, i, _ in extractor.extract_images(cbz)]))))

    # Prefetch: chapters come back in order, the next one is loaded while the caller works, never more than one ahead.
    saved_load = main.extractor.load_chapter
    loaded, in_flight, most = [], [0], [0]
    lock = threading.Lock()

    def slow_load(path, threads):
        with lock:
            in_flight[0] += 1
            most[0] = max(most[0], in_flight[0])
        time.sleep(0.05)
        loaded.append(path)
        with lock:
            in_flight[0] -= 1
        if path == "bad":
            raise ValueError("corrupt chapter")
        return [(f"{path}-page", path)]

    main.extractor.load_chapter = slow_load
    try:
        order, ahead = [], []
        for (path, _), pages in main.prefetched([(f"c{i}", None) for i in range(5)]):
            time.sleep(0.1)   # "upscaling": the next chapter should load meanwhile
            order.append(pages[0][1])
            ahead.append(len(loaded) - len(order))
        checks.append(("decode ahead: chapters arrive in order, each loaded while the previous one is processed",
                       order == ["c0", "c1", "c2", "c3", "c4"] and ahead == [1, 1, 1, 1, 0] and most[0] == 1))
        got = []
        try:
            for (path, _), pages in main.prefetched([("c0", None), ("bad", None), ("c2", None)]):
                got.append(path)
            raised = False
        except ValueError:
            raised = True
        checks.append(("decode ahead: a chapter that fails to load raises when its turn comes, after the ones before it",
                       raised and got == ["c0"]))
    finally:
        main.extractor.load_chapter = saved_load

    # Thread pools: a single process keeps its defaults; N processes each get max(1, cpu_count // N).
    saved_threads = (torch.get_num_threads(), cv2.getNumThreads(), main._decode_threads)
    try:
        one = main.configure_threads(1)
        unchanged = (torch.get_num_threads(), cv2.getNumThreads(), main._decode_threads) == saved_threads
        many = main.configure_threads(10 ** 6)
        share = (os.cpu_count() or 1) // 4
        torch.set_num_threads(saved_threads[0]); cv2.setNumThreads(saved_threads[1]); main._decode_threads = saved_threads[2]
        four = main.configure_threads(4)
        checks.append(("threads: --parallel 1 keeps torch, OpenCV and decode threads as they were", unchanged and one["share"] == (os.cpu_count() or 1)))
        checks.append(("threads: N processes each get at most cpu_count // N, and never less than 1",
                       many == {"share": 1, "torch": 1, "opencv": 1, "decode": 1}
                       and four["torch"] <= max(1, share) and four["opencv"] <= max(1, share) and four["decode"] <= main.DECODE_THREADS))
    finally:
        torch.set_num_threads(saved_threads[0]); cv2.setNumThreads(saved_threads[1]); main._decode_threads = saved_threads[2]

    saved_runtime = engine._runtime
    try:
        engine._runtime = engine._Runtime()
        engine._runtime.failed_shapes, engine._runtime.batch_override = {(512, 1500)}, {(512, 800): 2}
        state = engine.warm_up_state()
        engine._runtime = engine._Runtime()
        engine.adopt_warm_up(state["failed_shapes"], state["batch_override"])
        checks.append(("parallel: workers start from the warm-up's failed shapes and lowered batches",
                       engine._runtime.failed_shapes == {(512, 1500)} and engine._runtime.batch_override == {(512, 800): 2}))
    finally:
        engine._runtime = saved_runtime

    saved = (hdfyer_profile.cache.load_section, hdfyer_profile.profile_width)
    measured = []
    hdfyer_profile.cache.load_section = lambda key: {"weights": "w", "strip_height": 512, "widths": {"800": {"best_batch": 2}}}
    hdfyer_profile.profile_width = lambda *args, **kwargs: measured.append(args) or {"best_batch": 9}
    try:
        profile = hdfyer_profile.BatchProfile(None, "w", 512, on_missing=lambda width: 1)
        checks.append(("parallel: a worker never measures a missing width; it answers through on_missing",
                       profile(512, 800) == 2 and profile(512, 1500) == 1 and measured == []))
    finally:
        hdfyer_profile.cache.load_section, hdfyer_profile.profile_width = saved
    return checks


def conversion_checks() -> list[tuple[str, bool]]:
    """GPU-side pre/post-processing must match the CPU code it replaced, bit for bit. Runs on the CPU device."""

    def old_preprocess(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).half()

    def old_postprocess(tensor, round_output):
        scaled = tensor.clamp(0, 1).to(torch.float32).squeeze(0) * 255
        if round_output:
            scaled = scaled.round()
        return cv2.cvtColor(np.transpose(scaled.byte().numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR)

    checks = []
    saved_device = engine.DEVICE
    engine.DEVICE = torch.device("cpu")
    try:
        every_value = np.stack(np.meshgrid(np.arange(256), np.arange(3), indexing="ij"), -1)[..., :1].repeat(3, -1).astype(np.uint8)
        every_value[:, 1, 1] = every_value[::-1, 0, 0]   # vary the channels, so a channel swap would show
        new, old = engine._preprocess(every_value), old_preprocess(every_value)
        checks.append(("conversion: upload-then-convert is bit-identical to the CPU conversion for all 256 values, channel order included",
                       new.shape == old.shape and torch.equal(new.view(torch.int16), old.view(torch.int16))))
        y = (torch.rand((1, 3, 40, 24), generator=torch.Generator().manual_seed(5)) * 1.2 - 0.1).half()
        checks.append(("conversion: output conversion is byte-identical, truncating and rounding",
                       all(np.array_equal(engine._postprocess(y, r), old_postprocess(y, r)) for r in (False, True))
                       and engine._postprocess(y).flags["C_CONTIGUOUS"]))
    finally:
        engine.DEVICE = saved_device
    return checks


def pinned_pool_checks() -> list[tuple[str, bool]]:
    """The pinned-buffer pool reuses buffers once their copy is done and allocates only when nothing free fits."""

    class FakeEvent:
        def __init__(self, done):
            self.done = done

        def query(self):
            return self.done

    class CpuPool(engine._PinnedPool):
        @staticmethod
        def _allocate(nbytes):
            return torch.empty(nbytes, dtype=torch.uint8)   # stands in for pinned memory, which needs a GPU

    checks = []
    pool = CpuPool()
    a, view = pool.take((512, 800, 3))
    checks.append(("pinned pool: a buffer comes back as a view of the requested shape", tuple(view.shape) == (512, 800, 3)))
    pool.give(a, FakeEvent(done=False))
    b, _ = pool.take((512, 800, 3))
    checks.append(("pinned pool: a buffer whose copy is still running is not reused", b is not a and pool.allocations == 2))
    pool.give(b, FakeEvent(done=True))
    c, _ = pool.take((512, 800, 3))
    checks.append(("pinned pool: a finished buffer is reused instead of allocating", c is b and pool.allocations == 2))
    pool.give(c)
    d, _ = pool.take((300, 800, 3))
    checks.append(("pinned pool: a smaller shape reuses a bigger free buffer", d is c and pool.allocations == 2))
    pool.give(d)
    e, view = pool.take((512, 1500, 3))
    checks.append(("pinned pool: a larger shape grows the pool, drops finished buffers too small for it, keeps busy ones",
                   pool.allocations == 3 and tuple(view.shape) == (512, 1500, 3)
                   and [buffer is a for buffer, _ in pool._free] == [True]))
    pool.give(e)
    for _ in range(100):   # steady state: one strip at a time, its copy finished before the next
        buffer, _ = pool.take((512, 800, 3))
        pool.give(buffer, FakeEvent(done=True))
    checks.append(("pinned pool: steady state allocates nothing more", pool.allocations == 3))
    return checks


def recovery_checks() -> list[tuple[str, bool]]:
    """Out-of-memory and engine failures must be logged, retried smaller, and still produce the page. Runs on the CPU."""
    checks = []
    strip, batch = 64, 8
    image = np.random.default_rng(3).integers(0, 256, (300, 40, 3), dtype=np.uint8)
    expected = fake_upscale(image, STUDENT_SCALE)
    ALLOCATOR_ASSERT = '!handles_.at(i) INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":419, please report a bug to PyTorch.'

    def same(page, reference):
        # blending identical overlaps can round down by 1
        return page.shape == reference.shape and int(np.abs(page.astype(int) - reference.astype(int)).max()) <= 1

    def run_page(backend, model, fake_trt=None, page=image):
        """One page through chunk_page -> build_batches -> batch_upscale -> reassemble. Returns (page, events)."""
        runtime = engine._Runtime()
        runtime.backend, runtime.strip_height, runtime.round_output = backend, strip, True
        runtime.failed_shapes, runtime.batch_override, runtime.events = set(), {}, []
        runtime.trt, runtime.trt_batch, runtime.context = fake_trt, batch, "Chapter 1"
        engine._runtime = runtime
        chunks = chunk_page(page, 0, None, strip_height=strip)
        for chunk in chunks:
            chunk.page_name = "007.jpg"
        engine.prepare_chapter(chunks, STUDENT_SCALE)
        done = []
        for group in build_batches(chunks, None, batch):
            done.extend(engine.batch_upscale(group, model, STUDENT_SCALE))
        done.sort(key=lambda pair: pair[1].chunk_index)
        return reassemble(done, STUDENT_SCALE), engine.fallback_events()

    saved_runtime, saved_device = engine._runtime, engine.DEVICE
    engine.DEVICE = torch.device("cpu")
    try:
        page, events = run_page("eager", FakeOutOfMemory())
        checks.append(("recovery: a clean run gives the page and no fallback events",
                       same(page, expected) and events == []))

        model = FakeOutOfMemory(largest_batch=2)
        page, events = run_page("eager", model)
        checks.append(("recovery: eager out-of-memory is retried in smaller batches and the page still comes out",
                       same(page, expected) and max(model.calls) == 8 and model.calls.count(8) == 1
                       and {e["kind"] for e in events} == {"out of memory"}))
        checks.append(("recovery: the event names the chapter, the page, the strip shape and the reason",
                       bool(events) and events[0]["context"] == "Chapter 1" and events[0]["pages"] == ["007.jpg"]
                       and events[0]["shape"] == "64x40" and "out of memory" in events[0]["reason"]))

        model = FakeOutOfMemory(message=ALLOCATOR_ASSERT, fail_first=1)
        page, events = run_page("eager", model, page=image[:64])
        checks.append(("recovery: the allocator's internal assert is treated as out-of-memory and a single strip is retried once",
                       same(page, expected[:128]) and model.calls == [1, 1] and len(events) == 1))

        try:
            run_page("eager", FakeOutOfMemory(largest_batch=0), page=image[:64])
            raised = False
        except RuntimeError as e:
            raised = "out of memory" in str(e)
        checks.append(("recovery: a single strip that never fits still raises", raised))

        try:
            run_page("eager", FakeOutOfMemory(largest_batch=0, message="shape mismatch"))
            raised = False
        except RuntimeError as e:
            raised = "shape mismatch" in str(e)
        checks.append(("recovery: an eager error that is not out-of-memory is not swallowed", raised))

        eager = FakeOutOfMemory()
        trt = FakeTrt(FakeOutOfMemory(largest_batch=2))
        page, events = run_page("trt", eager, trt)
        checks.append(("recovery: TensorRT out-of-memory halves the engine batch until it fits, and the page still comes out",
                       same(page, expected) and [b for _, _, b in trt.built] == [8, 4, 2] and eager.calls == []
                       and [e["kind"] for e in events] == ["smaller batch", "smaller batch"] and trt.released >= 2))

        trt = FakeTrt(FakeOutOfMemory(), largest_build=2)
        page, events = run_page("trt", eager, trt)
        checks.append(("recovery: a failed engine build is retried with a smaller batch before the chapter starts",
                       same(page, expected) and [b for _, _, b in trt.built] == [2] and eager.calls == []
                       and [e["kind"] for e in events] == ["smaller batch", "smaller batch"]))

        eager = FakeOutOfMemory()
        trt = FakeTrt(FakeOutOfMemory(), largest_build=0)
        page, events = run_page("trt", eager, trt)
        checks.append(("recovery: when no engine can be built the shape goes to eager, loudly, and the page still comes out",
                       same(page, expected) and trt.built == [] and eager.calls != []
                       and [e["kind"] for e in events] == ["smaller batch"] * 3 + ["trt -> eager"]
                       and engine._runtime.failed_shapes == {(64, 40)}))

        eager = FakeOutOfMemory()
        trt = FakeTrt(FakeOutOfMemory(largest_batch=0, message="TensorRT execution failed"))
        page, events = run_page("trt", eager, trt)
        checks.append(("recovery: a TensorRT error that is not out-of-memory goes straight to eager and is logged once",
                       same(page, expected) and [e["kind"] for e in events] == ["trt -> eager"]
                       and "TensorRT execution failed" in events[0]["reason"] and events[0]["pages"] == ["007.jpg"]))

        # A --parallel worker may not build: a missing engine goes straight to eager, once, without trying smaller batches.
        eager = FakeOutOfMemory()
        trt = FakeTrt(FakeOutOfMemory())
        trt.allow_build = False

        def missing(x):
            raise trt_backend.EngineMissingError("no cached engine for 64x40 (batch 8) and this process may not build one")
        trt.run = missing
        page, events = run_page("trt", eager, trt)
        checks.append(("recovery: a worker with no cached engine falls back to eager at once, without building",
                       same(page, expected) and trt.built == [] and [e["kind"] for e in events] == ["trt -> eager"]
                       and "may not build" in events[0]["reason"]))

        # A strip that is not the standard shape has no engine: it runs eager and that is reported, every time.
        runtime = engine._Runtime()
        runtime.backend, runtime.strip_height, runtime.round_output = "trt", strip, True
        runtime.failed_shapes, runtime.batch_override, runtime.events = set(), {}, []
        runtime.trt, runtime.trt_batch = FakeTrt(FakeOutOfMemory()), 1
        engine._runtime = runtime
        odd = ChunkMeta(4, image[:50], 0, 1, 0, 50, 0, 0, page_name="credits.png")
        out = engine.batch_upscale([odd], FakeOutOfMemory(), STUDENT_SCALE)
        events = engine.fallback_events()
        checks.append(("recovery: an odd-shaped strip runs eager and is reported with its page",
                       np.array_equal(out[0][0], expected[:100]) and len(events) == 1 and events[0]["kind"] == "trt -> eager"
                       and events[0]["pages"] == ["credits.png"] and events[0]["shape"] == "50x40"))
        checks.append(("recovery: an event formats as one readable line",
                       engine.format_event(events[0]).startswith("[trt -> eager] pages credits.png | strips 50x40: ")))
    finally:
        engine._runtime, engine.DEVICE = saved_runtime, saved_device
    return checks


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
    checks.append(("cache key changes with batch", trt_backend.cache_key(**base, batch=8) != key))
    checks.append(("cache key changes between batch sizes above 1",
                   trt_backend.cache_key(**base, batch=8) != trt_backend.cache_key(**base, batch=4)))
    checks.append(("cache key changes with arch", trt_backend.cache_key(**base, arch="span") != key))
    checks.append(("batch 1 keeps the pre-batching engine name", trt_backend.cache_key(**base, batch=1, arch="rrdb") ==
                   "rrdb_512x800_4x-UltraSharp.pth-aaaaaaaaaaaa_nvidia-geforce-rtx-4060-laptop-gpu_trt11.4.0.106_fp16.plan"))
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
        checks.append(("hdfyer: every planned strip is 512 rows, short and two-strip pages included",
                       shapes == {(512, 800), (512, 690), (512, 1080)}))
        chunks = chunk_page(np.broadcast_to(np.zeros((1000, 1, 3), np.uint8), (1000, 800, 3)), 0, None, strip_height=STUDENT_STRIP_HEIGHT)
        checks.append(("hdfyer: the last strip is anchored to the page bottom and its extra overlap cropped",
                       [(c.y_start, c.y_end, c.overlap_top, c.overlap_bottom, c.crop_top, c.crop_bottom) for c in chunks]
                       == [(0, 512, 0, 32, 0, 0), (480, 992, 32, 32, 0, 0), (960, 1000, 32, 0, 472, 0)]))
        chunks = chunk_page(np.broadcast_to(np.arange(300, dtype=np.uint8).reshape(300, 1, 1), (300, 800, 3)), 0, None, strip_height=STUDENT_STRIP_HEIGHT)
        checks.append(("hdfyer: a short page is reflect-padded to one strip and cropped back",
                       len(chunks) == 1 and chunks[0].page_slice.shape[:2] == (512, 800) and chunks[0].crop_bottom == 212
                       and (chunks[0].y_start, chunks[0].y_end) == (0, 300)
                       and chunks[0].page_slice[300:303, 0, 0].tolist() == [298 % 256, 297 % 256, 296 % 256]))
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
                   [len(b) for b in batches if b[0].page_slice.shape[:2] == (512, 800)] == [8, 8, 5]))
    checks.append(("batching: strips of different pages share a batch",
                   any(len({c.page_id for c in b}) > 1 for b in batches)))
    checks.append(("batching: batch size 1 gives one strip per batch",
                   all(len(b) == 1 for b in build_batches(chunks, None, batch_size=1))))

    batches = build_batches(chunks, None, batch_size=lambda h, w: 6 if w == 800 else 3)
    checks.append(("batching: a per-shape batch size is applied to each shape",
                   max(len(b) for b in batches if b[0].page_slice.shape[:2] == (512, 800)) == 6
                   and max(len(b) for b in batches if b[0].page_slice.shape[:2] == (512, 690)) == 3))

    saved_runtime = engine._runtime
    try:
        runtime = engine._Runtime()
        runtime.batch_override = {}
        engine._runtime = runtime
        checks.append(("engine batch defaults to 1 (teacher)", engine._engine_batch(512, 800) == 1))
        runtime.trt_batch = lambda h, w: 6 if w == 800 else 3
        checks.append(("engine batch follows the per-shape batch size", engine._engine_batch(512, 800) == 6
                       and engine._engine_batch(512, 1500) == 3))
        runtime.batch_override[(512, 800)] = 2
        checks.append(("engine batch lowered after an out-of-memory wins over the configured one",
                       engine._engine_batch(512, 800) == 2 and engine._engine_batch(512, 1500) == 3))
    finally:
        engine._runtime = saved_runtime

    checks += profile_checks()
    checks += parallel_checks()
    checks += conversion_checks()
    checks += pinned_pool_checks()
    checks += recovery_checks()

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
