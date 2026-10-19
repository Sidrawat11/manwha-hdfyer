"""digest: strips and stitching, engine routing and fallbacks, conversions, pinned buffers, batch profile, the model's fold."""

import json
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from hdfyer.ingest import reader as extractor
from hdfyer.digest.model import fold_reparam
from hdfyer.digest import engine, trt as trt_backend
from hdfyer.digest.strips import ChunkMeta, chunk_page, crop_upscaled
from hdfyer.digest.batches import build_batches
from hdfyer.digest import batch_profile as hdfyer_profile
from hdfyer.digest.stitch import reassemble
from tests.helpers import *


def round_trip_checks(cbz_path: Path | None = None) -> list[tuple[str, bool]]:
    """Strips + stitching: every real page of a CBZ, a synthetic page, and a sweep of page sizes come back pixel-exact."""
    results = []
    if cbz_path is None:
        found = sorted(Path("data/library").rglob("*.cbz"))
        cbz_path = found[0] if found else None
    small = np.random.default_rng(0).integers(0, 256, (400, 300, 3), dtype=np.uint8)
    cases = [("synthetic 300x400", small)]
    if cbz_path is not None:
        cases += [(name, image) for name, image, _ in extract_images(cbz_path)]
    else:
        print("  (no CBZ in data/library/: real-page cases skipped)")

    print(f"  round trip: {len(cases) - 1} pages from {cbz_path}")
    for name, image in cases:
        try:
            n, diff = round_trip(image, fake_upscale_2x, STUDENT_SCALE, STUDENT_STRIP_HEIGHT)
            ok = diff <= 1  # blending identical overlaps can round down by 1
            print(f"  {'OK  ' if ok else 'FAIL'} {name}: {image.shape[1]}x{image.shape[0]}, {n} chunks, max diff {diff}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL {name}: {e}")
        results.append((f"round trip: {name}", ok))

    print(f"  sweep: widths {SWEEP_WIDTHS}, heights {SWEEP_HEIGHTS.start}-{SWEEP_HEIGHTS[-1]} in steps of {SWEEP_HEIGHTS.step}, plus short and boundary heights")
    rng = np.random.default_rng(1)
    heights = list(SWEEP_HEIGHTS) + SHORT_HEIGHTS
    for w in SWEEP_WIDTHS:
        failed, worst, most_chunks = 0, 0, 0
        for h in heights:
            try:
                n, diff = round_trip(sweep_page(rng, w, h), fake_upscale_column_2x, STUDENT_SCALE, STUDENT_STRIP_HEIGHT)
                ok = diff <= 1
                worst, most_chunks = max(worst, diff), max(most_chunks, n)
                if not ok:
                    print(f"  FAIL hdfyer {w}x{h}: {n} chunks, max diff {diff}")
            except AssertionError as e:
                ok = False
                print(f"  FAIL hdfyer {w}x{h}: {e}")
            failed += not ok
            results.append((f"round trip: {w}x{h} sweep page", ok))
        print(f"  {'OK  ' if not failed else 'FAIL'} hdfyer strips, width {w}: {len(heights) - failed}/{len(heights)} heights, "
              f"up to {most_chunks} chunks, max diff {worst}")
    return results


def profile_checks() -> list[tuple[str, bool]]:
    """hdfyer's batch profile: which batch is chosen, and that it is measured once and then read from the cache."""
    checks = []
    pick = hdfyer_profile.pick_best_batch
    checks.append(("profile: equal time per strip picks the smallest batch", pick({1: 17.6, 2: 17.7, 4: 17.5, 8: 17.8}) == 1))
    checks.append(("profile: a clearly faster batch is picked", pick({1: 20.0, 2: 15.0, 4: 12.0, 8: 11.9}) == 4))
    checks.append(("profile: a gain under 3% does not justify a bigger batch", pick({1: 10.0, 2: 9.8}) == 1 and pick({1: 10.0, 2: 9.6}) == 2))
    checks.append(("profile: a single measurement is its own best", pick({1: 58.0}) == 1))

    saved = (hdfyer_profile.load_section, hdfyer_profile.save_section, hdfyer_profile.profile_width)
    stored, measured, freed = {}, [], []

    def fake_profile_width(model, width, strip_height, vram_mb=None):
        measured.append(width)
        return {"best_batch": 4 if width >= 1000 else 2, "batches": {"1": {"ms_per_strip": 9.0, "peak_vram_mb": 100}}, "stopped": "test"}

    hdfyer_profile.load_section = lambda key: dict(stored.get(key, {}))
    hdfyer_profile.save_section = lambda key, section: stored.__setitem__(key, json.loads(json.dumps(section)))
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
        hdfyer_profile.load_section, hdfyer_profile.save_section, hdfyer_profile.profile_width = saved
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
        chunks = chunk_page(page, 0, strip)
        for chunk in chunks:
            chunk.page_name = "007.jpg"
        engine.prepare_chapter(chunks, STUDENT_SCALE)
        done = []
        for group in build_batches(chunks, batch):
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


def engine_checks() -> list[tuple[str, bool]]:
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

    # Routing: only the standard strip shape goes to the backend, everything else is eager.
    saved_runtime = engine._runtime
    try:
        strip = STUDENT_STRIP_HEIGHT
        for backend in ("trt", "compiled"):
            runtime = engine._Runtime()
            runtime.backend, runtime.strip_height, runtime.failed_shapes = backend, strip, set()
            engine._runtime = runtime
            for w in SWEEP_WIDTHS:
                checks.append((f"{backend}: standard {w}x{strip} strip uses {backend}", engine.route(strip, w) == backend))
                others = [strip - 1, strip + 1, strip + 16, 130, 400]
                checks.append((f"{backend}: non-standard heights at width {w} run eager",
                               all(engine.route(h, w) == "eager" for h in others if h != strip)))
            runtime.failed_shapes.add((strip, 800))
            checks.append((f"{backend}: a shape that failed runs eager", engine.route(strip, 800) == "eager"))
            checks.append((f"{backend}: other shapes keep {backend} after a failure", engine.route(strip, 690) == backend))

        runtime = engine._Runtime()
        runtime.backend, runtime.strip_height, runtime.failed_shapes = "eager", strip, set()
        engine._runtime = runtime
        checks.append(("eager: standard strips run eager", engine.route(strip, 800) == "eager"))
    finally:
        engine._runtime = saved_runtime

    # hdfyer: one fixed strip height for every width.
    saved_runtime = engine._runtime
    try:
        runtime = engine._Runtime()
        runtime.backend, runtime.failed_shapes = "trt", set()
        runtime.strip_height = STUDENT_STRIP_HEIGHT
        engine._runtime = runtime
        checks.append(("student: 512-row strips use the backend at any width",
                       all(engine.route(STUDENT_STRIP_HEIGHT, w) == "trt" for w in (436, 690, 800, 1080, 3200))))
        checks.append(("student: other heights run eager",
                       all(engine.route(h, 800) == "eager" for h in (130, 379, 511, 513, 528, 898))))
        shapes = set()
        for w, h in [(800, 300), (800, 512), (800, 1000), (800, 4540), (690, 3000), (1080, 9000)]:
            page = np.broadcast_to(np.zeros((h, 1, 3), np.uint8), (h, w, 3))
            shapes |= {c.page_slice.shape[:2] for c in chunk_page(page, 0, STUDENT_STRIP_HEIGHT)}
        checks.append(("hdfyer: every planned strip is 512 rows, short and two-strip pages included",
                       shapes == {(512, 800), (512, 690), (512, 1080)}))
        chunks = chunk_page(np.broadcast_to(np.zeros((1000, 1, 3), np.uint8), (1000, 800, 3)), 0, STUDENT_STRIP_HEIGHT)
        checks.append(("hdfyer: the last strip is anchored to the page bottom and its extra overlap cropped",
                       [(c.y_start, c.y_end, c.overlap_top, c.overlap_bottom, c.crop_top, c.crop_bottom) for c in chunks]
                       == [(0, 512, 0, 32, 0, 0), (480, 992, 32, 32, 0, 0), (960, 1000, 32, 0, 472, 0)]))
        chunks = chunk_page(np.broadcast_to(np.arange(300, dtype=np.uint8).reshape(300, 1, 1), (300, 800, 3)), 0, STUDENT_STRIP_HEIGHT)
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
        chunks += chunk_page(page, page_id, STUDENT_STRIP_HEIGHT)
    batches = build_batches(chunks, batch_size=8)
    checks.append(("batching: every chunk is in exactly one batch",
                   sorted(id(c) for b in batches for c in b) == sorted(id(c) for c in chunks)))
    checks.append(("batching: a batch holds one shape and at most 8 strips",
                   all(len(b) <= 8 and len({c.page_slice.shape for c in b}) == 1 for b in batches)))
    checks.append(("batching: only the last batch of a shape is short",
                   [len(b) for b in batches if b[0].page_slice.shape[:2] == (512, 800)] == [8, 8, 5]))
    checks.append(("batching: strips of different pages share a batch",
                   any(len({c.page_id for c in b}) > 1 for b in batches)))
    checks.append(("batching: batch size 1 gives one strip per batch",
                   all(len(b) == 1 for b in build_batches(chunks, batch_size=1))))

    batches = build_batches(chunks, batch_size=lambda h, w: 6 if w == 800 else 3)
    checks.append(("batching: a per-shape batch size is applied to each shape",
                   max(len(b) for b in batches if b[0].page_slice.shape[:2] == (512, 800)) == 6
                   and max(len(b) for b in batches if b[0].page_slice.shape[:2] == (512, 690)) == 3))

    saved_runtime = engine._runtime
    try:
        runtime = engine._Runtime()
        runtime.batch_override = {}
        engine._runtime = runtime
        checks.append(("engine batch defaults to 1", engine._engine_batch(512, 800) == 1))
        runtime.trt_batch = lambda h, w: 6 if w == 800 else 3
        checks.append(("engine batch follows the per-shape batch size", engine._engine_batch(512, 800) == 6
                       and engine._engine_batch(512, 1500) == 3))
        runtime.batch_override[(512, 800)] = 2
        checks.append(("engine batch lowered after an out-of-memory wins over the configured one",
                       engine._engine_batch(512, 800) == 2 and engine._engine_batch(512, 1500) == 3))
    finally:
        engine._runtime = saved_runtime


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




def checks(cbz_path: Path | None = None) -> list[tuple[str, bool]]:
    return (round_trip_checks(cbz_path) + engine_checks() + profile_checks() + conversion_checks()
            + pinned_pool_checks() + recovery_checks())
