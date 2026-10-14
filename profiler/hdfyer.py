"""Batch-size profile for hdfyer.

For each page width, times a forward pass over 512-row strips at batch 1, 2, 4
and 8 (stopping at a VRAM cap) and records the batch with the best time per
strip. Results live in the GPU's cache/<gpu>.json under "hdfyer", next to the
teacher's benchmark map, and are reused on later runs. A width is profiled the
first time it is needed, which takes a few seconds.

The measurement is eager FP16. That is the runtime with the largest memory use
(the fallback when a TensorRT engine fails), so a batch that fits here is safe
for both.
"""

import gc
import time
from pathlib import Path
from typing import Callable
import torch
from tqdm import tqdm
from profiler import cache

SECTION = "hdfyer"
BATCHES = (1, 2, 4, 8)
VRAM_CAP = 0.75        # share of total VRAM a batch's forward pass may reach
MIN_GAIN = 0.03        # a larger batch must be at least this much faster per strip to be chosen
WARMUP_PASSES = 2
TIMED_PASSES = 5


def pick_best_batch(ms_per_strip: dict[int, float], min_gain: float = MIN_GAIN) -> int:
    """
    The smallest batch whose time per strip is within min_gain of the fastest.
    Smaller batches use less VRAM and their TensorRT engines build faster, so a larger one has to earn its place.
    """
    fastest = min(ms_per_strip.values())
    return min(batch for batch, ms in ms_per_strip.items() if ms <= fastest * (1 + min_gain))


def _is_oom(error: Exception) -> bool:
    message = str(error)
    return (isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower()
            or ("INTERNAL ASSERT FAILED" in message and "CUDACachingAllocator" in message))


def _measure(model: torch.nn.Module, batch: int, height: int, width: int) -> dict | None:
    """Median forward time and peak VRAM for one batch size. None if it runs out of memory."""
    gc.collect()
    torch.cuda.empty_cache()
    failed = False
    try:
        x = torch.rand((batch, 3, height, width), dtype=torch.float16, device="cuda")
        with torch.no_grad():
            for _ in range(WARMUP_PASSES):   # cuDNN autotuning happens here, outside the timing
                model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(TIMED_PASSES):
                start = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    except Exception as e:
        if not _is_oom(e):
            raise
        failed = True

    # Outside the except block, so the failed pass's tensors are no longer held by the traceback.
    x = None
    gc.collect()
    torch.cuda.empty_cache()
    if failed:
        return None
    timings.sort()
    return {"ms_per_strip": round(timings[len(timings) // 2] * 1000 / batch, 2), "peak_vram_mb": round(peak_mb)}


def profile_width(model: torch.nn.Module, width: int, strip_height: int, vram_mb: float | None = None) -> dict:
    """Measures batch 1, 2, 4, 8 for strips of one width, stopping before a batch that would pass the VRAM cap."""
    if vram_mb is None:
        vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    budget_mb = VRAM_CAP * vram_mb
    measured = {}
    stopped = "all batch sizes measured"

    for batch in BATCHES:
        if measured:
            previous = max(measured)
            predicted_mb = measured[previous]["peak_vram_mb"] * batch / previous
            if predicted_mb > budget_mb:
                stopped = f"batch {batch} would need about {predicted_mb:.0f} MB, over the {budget_mb:.0f} MB cap"
                break
        result = _measure(model, batch, strip_height, width)
        if result is None:
            stopped = f"batch {batch} ran out of memory"
            break
        if batch > 1 and result["peak_vram_mb"] > budget_mb:
            stopped = f"batch {batch} peaked at {result['peak_vram_mb']} MB, over the {budget_mb:.0f} MB cap"
            break
        measured[batch] = result

    if not measured:
        # Not even one strip fits: batch 1 is still the only thing to try, and the engine's own recovery takes it from there.
        return {"best_batch": 1, "batches": {}, "stopped": stopped}
    best = pick_best_batch({batch: result["ms_per_strip"] for batch, result in measured.items()})
    return {"best_batch": best, "batches": {str(batch): result for batch, result in measured.items()}, "stopped": stopped}


class BatchProfile:
    """
    hdfyer's best batch size per page width, read from the GPU's profile and measured when missing.
    Call it with a strip's (height, width); odd widths count as the even width the engine pads them to.
    """

    def __init__(self, model: torch.nn.Module, weights_id: str, strip_height: int, before_measuring: Callable[[], None] | None = None,
                 on_missing: Callable[[int], int] | None = None):
        """on_missing(width) -> batch, if given, answers for a width with no profile instead of measuring it
        (--parallel workers: the warm-up measured every width, and workers must not write the cache)."""
        self._model = model
        self._strip_height = strip_height
        self._before_measuring = before_measuring
        self._on_missing = on_missing
        section = cache.load_section(SECTION)
        # A profile for other weights or another strip height says nothing about this model.
        if section.get("weights") != weights_id or section.get("strip_height") != strip_height:
            section = {"weights": weights_id, "strip_height": strip_height, "runtime": "eager fp16",
                       "vram_cap": VRAM_CAP, "widths": {}}
        self._section = section

    def __call__(self, height: int, width: int) -> int:
        width += width % 2
        entry = self._section["widths"].get(str(width))
        if entry is None:
            if self._on_missing is not None:
                return self._on_missing(width)
            entry = self._measure(width)
        return entry["best_batch"]

    def _measure(self, width: int) -> dict:
        tqdm.write(f"Profiling hdfyer batch size for {width} px wide pages (one-time for this GPU)...")
        if self._before_measuring is not None:
            self._before_measuring()   # e.g. free a live TensorRT engine, so the measurement has the whole card
        entry = profile_width(self._model, width, self._strip_height)
        self._section["widths"][str(width)] = entry
        self._section["profiled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cache.save_section(SECTION, self._section)
        timings = ", ".join(f"batch {b}: {r['ms_per_strip']} ms/strip" for b, r in entry["batches"].items())
        tqdm.write(f"  best batch {entry['best_batch']} ({timings}; {entry['stopped']})")
        return entry
