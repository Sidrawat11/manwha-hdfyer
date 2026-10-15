"""Measurement loop for GPU profiling.

Runs warmup and timed forward passes per candidate shape, handles OOM gracefully,
and records peak VRAM and timing data.
"""

from pathlib import Path
import torch
import time
from hdfyer.digest.model import load_rrdbnet
import torch.backends.cudnn as cudnn
import math
cudnn.benchmark = True

CANDIDATE_HEIGHTS = [128, 256, 512, 1024, 2048]  # base list, max_height appended at runtime
DEVICE = torch.device('cuda')
DTYPE = torch.float16


def _format_elapsed(seconds: float) -> str:
    """Formats seconds into 'Xm Ys' string."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:.0f}m {secs:.0f}s"


def _build_candidate_shapes(unique_widths: set[int], max_height: int) -> list[tuple[int, int]]:
    """Crosses unique widths against candidate heights and returns shapes sorted largest-first."""
    results = []
    for w in unique_widths:
        for h in CANDIDATE_HEIGHTS:
            results.append((w, h))
        results.append((w, max_height))

    return sorted(results, key=lambda shape: (-shape[1], shape[0]))


def _profile_shape(w: int, h: int, model: torch.nn.Module) -> dict:
    """
    Profiles a single shape. 
    Creates synthetic tensor, runs warmup and 5 timed forward passes, records peak VRAM. Handles OOM and cleanup internally. 
    Returns dict with avg_time_ms, peak_vram_mb, and valid.
    """
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Pre-Flight check only for windows to avoid "Thrashing"
        estimated_vram = estimate_shape_vram(h, w)
        available_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2

        # If estimated VRAM exceeds 75% of available VRAM we never touch cuda.
        # Its a generous estimate making sure 25% is available for activations.
        if estimated_vram > available_vram * 0.75:
            return {
                "width" : w,
                "height" : h,
                "valid": False,
                "reason": "OOM"
            }
        
        x = torch.rand((1, 3, h, w), dtype=DTYPE, device=DEVICE)

        # Drain any previous tensors in the memory.
        torch.cuda.synchronize()

        # Run warm-ups so cuDNN bechmark doesn't affect the timing calculations
        with torch.no_grad():
            for _ in range(2):
                warmup_start = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
                warmup_elapsed = time.perf_counter() - warmup_start
                print(f"    Warmup pass took {warmup_elapsed:.1f}s")

        # Reset the VRAM memory.
        torch.cuda.reset_peak_memory_stats()

        # Run the real timing forward-passes.
        with torch.no_grad():
            timings = []
            for _ in range(5):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                y = model(x)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                timings.append(t2 - t1)
        
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_bytes / 1024 ** 2
        
        timings.sort()
        mean_seconds = timings[len(timings) // 2]
       
        del x, y
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return {
            "width" : w,
            "height" : h,
            "avg_time_ms": mean_seconds * 1000, 
            "peak_vram_mb": peak_vram_mb, 
            "valid": True
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
            raise  # re-raise if it's a RuntimeError that's NOT OOM
        if 'x' in locals():
            del x
        torch.cuda.empty_cache()
        return {
                "width" : w,
                "height" : h,
                "valid": False,
                "reason": "OOM"
            }

def run_profiler(unique_widths: set[int], max_height: int, model_path: Path) -> dict:
    """
    Orchestrates the profiling sweep. 
    Loads model, builds candidate shapes, loops with oom_heights tracking, delegates measurement to _profile_shape. 
    No direct CUDA calls.
    """
    model = load_rrdbnet(model_path)
    shapes = _build_candidate_shapes(unique_widths, max_height)
    oom_height = set()
    entries = []
    
    for w, h in shapes:
        if h in oom_height:
            print(f"  Skipping {w}x{h} — height {h} previously OOMed")
            continue
        print(f"  Profiling {w}x{h}...")
        shape_start = time.perf_counter()
        entry = _profile_shape(w, h, model)
        elapsed = time.perf_counter() - shape_start
        if entry["valid"]:
            print(f"  Done {w}x{h} in {_format_elapsed(elapsed)}")
        else:
            print(f"  OOM {w}x{h} after {_format_elapsed(elapsed)}")
            oom_height.add(h)
        entries.append(entry)

    return {
        "page_widths_seen": list(unique_widths),
        "max_page_height": max_height,
        "entries": entries
    }

def estimate_shape_vram(h: int, w: int) -> float:
    return 0.0085 * (h * w) + 32.1