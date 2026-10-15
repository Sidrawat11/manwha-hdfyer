"""Strips: cutting pages into the fixed-shape pieces the model runs on.

Every strip is exactly strip_height rows, whatever the page. Strips step down the page
with a 32-row overlap and the last one is anchored to the page bottom; a page shorter
than a strip is reflect-padded up to one. The rows a strip carries beyond its share
(crop_top, crop_bottom) are cut from its upscaled output before stitching. One strip
shape per page width means one TensorRT engine per width covers every strip.
"""

from dataclasses import dataclass
import numpy as np

OVERLAP = 32   # rows shared by neighbouring strips; their upscaled overlaps are blended when stitching


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
    # Rows of page_slice outside y_start..y_end, to be cut from the upscaled strip (see crop_upscaled).
    crop_top: int = 0
    crop_bottom: int = 0
    page_name: str = ""


def crop_upscaled(upscaled: np.ndarray, chunk: ChunkMeta, scale: int) -> np.ndarray:
    """Cuts the rows a fixed-shape strip carried beyond its y_start..y_end span from its upscaled output."""
    if not chunk.crop_top and not chunk.crop_bottom:
        return upscaled
    return upscaled[chunk.crop_top * scale : upscaled.shape[0] - chunk.crop_bottom * scale]


def _fixed_chunks(image: np.ndarray, page_id: int, strip_height: int, overlap: int) -> list[ChunkMeta]:
    """Strips of exactly strip_height rows: stepped by strip_height - overlap, the last anchored to the page bottom."""
    h = image.shape[0]

    if h < strip_height:
        # Too short for a strip: reflect the page downwards to full height, and cut the reflection off afterwards.
        padded = np.pad(image, ((0, strip_height - h), (0, 0), (0, 0)), mode="reflect")
        return [ChunkMeta(page_id, padded, 0, 1, 0, h, 0, 0, crop_bottom=strip_height - h)]

    starts = []
    y = 0
    while y + strip_height < h:
        starts.append(y)
        y += strip_height - overlap

    total_chunks = len(starts) + 1
    results = []
    for ch_index, y_start in enumerate(starts):
        chunk = image[y_start : y_start + strip_height, :]
        results.append(ChunkMeta(page_id, chunk, ch_index, total_chunks, y_start, y_start + strip_height, overlap if ch_index > 0 else 0, overlap))

    # The last strip's share starts at y, but it is cut from the bottom strip_height rows so its shape is standard.
    # It overlaps its neighbour by more than `overlap`; the extra rows on top are discarded.
    anchored = h - strip_height
    results.append(ChunkMeta(page_id, image[anchored:, :], len(starts), total_chunks, y, h, overlap if starts else 0, 0,
                             crop_top=y - anchored))
    return results


def chunk_page(image: np.ndarray, page_id: int, strip_height: int, overlap: int = OVERLAP) -> list[ChunkMeta]:
    """
    Slices a page into strips whose page_slice is exactly strip_height rows (see _fixed_chunks).
    The upscaled strips need crop_upscaled before they are stitched.
    """
    return _fixed_chunks(image, page_id, strip_height, overlap)
