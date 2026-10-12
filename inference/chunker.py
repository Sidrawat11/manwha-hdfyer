"""Variable-selection chunking strategy.

Queries the benchmark map to determine the largest safe chunk height per page
per machine, then slices pages into GPU-ready chunks with overlap for feathered
stitching.
"""

from dataclasses import dataclass
import numpy as np
from profiler.runner import estimate_shape_vram
from profiler.benchmark import get_max_chunk_height

DEFAULT_OVERLAP = 128


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

def chunk_page(image: np.ndarray, page_id: int, benchmark_map: dict, overlap: int = DEFAULT_OVERLAP) -> list[ChunkMeta]:
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