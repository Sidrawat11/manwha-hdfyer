"""Page chunking strategy.

Slices pages into GPU-ready strips with overlap for feathered stitching.

Default ("smarter strips"): every strip of a page has the same height, starts are
spread evenly down the page, and neighbours overlap by about 32 rows. One shape
per page width means cuDNN tunes once and strips from any page batch together.

Legacy: the tallest strip the benchmark map allows, a fixed 128-row overlap and
a short last strip whose height changes with every page.

Fixed height (strip_height=...): the same planner with one strip height for every
width and no VRAM model. For models the benchmark map was not profiled for.
"""

import math
from dataclasses import dataclass
import numpy as np
from profiler.runner import estimate_shape_vram
from profiler.benchmark import get_max_chunk_height

DEFAULT_OVERLAP = 128      # legacy strips
SMART_OVERLAP = 32
SMART_STRIP_HEIGHT = 512   # rows for a SMART_REF_WIDTH-wide page; scaled to keep the same pixels per strip
SMART_REF_WIDTH = 800
MIN_STRIP_HEIGHT = 64


@dataclass
class ChunkMeta:
    page_id: int
    page_slice: np.ndarray
    chunk_index: int
    total_chunks: int
    y_start: int
    y_end: int
    overlap_top: int
    overlap_bottom: int


def get_strip_height(width: int, benchmark_map: dict) -> int:
    """Strip height for the smarter-strips planner: same pixel count per strip at any width, never above the VRAM limit."""
    height = min(SMART_STRIP_HEIGHT * SMART_REF_WIDTH // width, get_max_chunk_height(width, benchmark_map))
    return max(MIN_STRIP_HEIGHT, height)


def plan_strips(h: int, strip_height: int, overlap: int = SMART_OVERLAP, max_strip_height: int | None = None) -> list[tuple[int, int]]:
    """
    Returns (y_start, y_end) for strips of one fixed height, starts spread evenly down the page.
    Neighbours overlap by about `overlap` rows, and no strip's top + bottom overlap exceeds its height:
    the reassembler would duplicate or drop rows if it did.
    A page just under two strips tall gets two taller strips (up to max_strip_height) to keep the full overlap.
    """
    if h <= strip_height:
        return [(0, h)]

    def build(n: int) -> list[tuple[int, int]]:
        starts = [round(k * (h - strip_height) / (n - 1)) for k in range(n)]
        return [(y, y + strip_height) for y in starts]

    def fits(plan: list[tuple[int, int]]) -> bool:
        for k in range(1, len(plan) - 1):
            overlap_top = plan[k - 1][1] - plan[k][0]
            overlap_bottom = plan[k][1] - plan[k + 1][0]
            if overlap_top + overlap_bottom > strip_height:
                return False
        return True

    n = math.ceil((h - strip_height) / (strip_height - overlap)) + 1
    plan = build(n)
    while not fits(plan) and n > 2:
        n -= 1
        plan = build(n)

    # Dropping to two strips can leave them barely overlapping; grow both until they overlap properly.
    if len(plan) == 2 and plan[0][1] - plan[1][0] < overlap:
        taller = math.ceil((h + overlap) / 2)
        if max_strip_height is not None:
            taller = min(taller, max_strip_height)
        taller = max(taller, strip_height)
        plan = [(0, taller), (h - taller, h)]

    return plan


def _smart_chunks(image: np.ndarray, page_id: int, benchmark_map: dict, overlap: int, strip_height: int | None = None) -> list[ChunkMeta]:
    h, w = image.shape[:2]

    if strip_height is not None:
        # Fixed strip height: the benchmark map (profiled for the teacher) plays no part.
        plan = plan_strips(h, strip_height, overlap)
    else:
        # As in legacy: a page that fits the VRAM budget is upscaled in one pass.
        if estimate_shape_vram(h, w) < benchmark_map["available_vram"] * 0.75:
            return [ChunkMeta(page_id, image, 0, 1, 0, h, 0, 0)]

        plan = plan_strips(h, get_strip_height(w, benchmark_map), overlap, get_max_chunk_height(w, benchmark_map))
    total_chunks = len(plan)
    results = []

    for ch_index, (y_start, y_end) in enumerate(plan):
        overlap_top = plan[ch_index - 1][1] - y_start if ch_index > 0 else 0
        overlap_bottom = y_end - plan[ch_index + 1][0] if ch_index < total_chunks - 1 else 0
        chunk = image[y_start : y_end, :]
        results.append(ChunkMeta(page_id, chunk, ch_index, total_chunks, y_start, y_end, overlap_top, overlap_bottom))

    return results


def _legacy_chunks(image: np.ndarray, page_id: int, benchmark_map: dict, overlap: int) -> list[ChunkMeta]:
    h, w = image.shape[:2]

    available_vram = benchmark_map["available_vram"]
    estimated_vram = estimate_shape_vram(h, w)
    results = []

    # Pre-flight Check if the full page fits in the upscaler no chunking required.
    if estimated_vram < available_vram * 0.75:
        return [ChunkMeta(page_id, image, 0, 1, 0, h, 0, 0)]

    max_chunk_height = get_max_chunk_height(w, benchmark_map)
    ch_index = 0
    y_start = 0

    while y_start < h:
        y_end = y_start + max_chunk_height
        overlap_top = overlap if y_start > 0 else 0
        if y_end > h:
            chunk = image[y_start:,:]
            results.append(ChunkMeta(page_id, chunk, ch_index, 0, y_start, h, overlap_top, 0))
            break

        chunk = image[y_start : y_end, :]

        if y_start == 0:
            results.append(ChunkMeta(page_id, chunk, ch_index, 0, y_start, y_end, 0, overlap))
        else:
            results.append(ChunkMeta(page_id, chunk, ch_index, 0, y_start, y_end, overlap_top, overlap))
        
        y_start = y_end - overlap
        ch_index += 1
    
    total_chunks = len(results)
    for chunk in results:
        chunk.total_chunks = total_chunks
    
    return results


def chunk_page(image: np.ndarray, page_id: int, benchmark_map: dict | None, overlap: int | None = None, legacy_strips: bool = False,
               strip_height: int | None = None) -> list[ChunkMeta]:
    """
    Slices a page into strips. `overlap` defaults to 32 rows, or 128 with legacy_strips.
    With `strip_height` every strip is that tall whatever the width, and benchmark_map is not used.
    """
    if legacy_strips:
        return _legacy_chunks(image, page_id, benchmark_map, DEFAULT_OVERLAP if overlap is None else overlap)
    return _smart_chunks(image, page_id, benchmark_map, SMART_OVERLAP if overlap is None else overlap, strip_height)
