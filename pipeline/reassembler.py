"""Chunk stitcher.

Joins the upscaled chunks of one page back into a full page, linearly blending
each overlap region to hide seams.
"""

import numpy as np
from inference.chunker import ChunkMeta


def _blend_overlap(top_region: np.ndarray, bottom_region: np.ndarray) -> np.ndarray:
    """Linear alpha blend over the overlap region between two adjacent chunks."""
    alpha = np.linspace(0, 1, top_region.shape[0])
    alpha = alpha[:, np.newaxis, np.newaxis]
    blended = top_region * (1 - alpha) + bottom_region * alpha
    return blended.astype(np.uint8)

def _check_height(stitched: np.ndarray, expected_height: int, chunks: list[tuple[np.ndarray, ChunkMeta]]) -> None:
    """Raises if the stitched page isn't as tall as the span its chunks cover."""
    # Rows are silently duplicated or dropped when a chunk's overlap_top + overlap_bottom exceeds its height.
    if stitched.shape[0] != expected_height:
        meta = chunks[0][1]
        raise AssertionError(
            f"reassemble: page {meta.page_id} stitched to {stitched.shape[0]} rows, expected {expected_height} "
            f"({len(chunks)} chunks spanning rows {meta.y_start}-{chunks[-1][1].y_end}). "
            "Check that no chunk has overlap_top + overlap_bottom greater than its height."
        )


def reassemble(chunks: list[tuple[np.ndarray, ChunkMeta]], scale: int = 4) -> np.ndarray:
    """Stitches upscaled chunks into a full page with overlap blending. Receives paired upscaled arrays and their ChunkMeta, returns the reconstructed page."""
    chunk_body, chunk_meta = chunks[0]
    expected_height = (chunks[-1][1].y_end - chunk_meta.y_start) * scale
    # Unchunked page: overlap_bottom is 0, and [:-0] would slice to an empty array.
    if len(chunks) == 1:
        _check_height(chunk_body, expected_height, chunks)
        return chunk_body
    pieces = [chunk_body[:-chunk_meta.overlap_bottom*scale]]
    for i in range(1, len(chunks)):
        chunk_A, meta_A = chunks[i - 1]
        chunk_B, meta_B = chunks[i]

        blended = _blend_overlap(chunk_A[-meta_A.overlap_bottom*scale:], chunk_B[:meta_B.overlap_top*scale])

        if i == len(chunks) - 1:
            img_B = chunk_B[meta_B.overlap_top*scale:]
        else:
            img_B = chunk_B[meta_B.overlap_top*scale:-meta_B.overlap_bottom*scale]

        pieces.extend([blended, img_B])
    
    stitched = np.concatenate(pieces, axis=0)
    _check_height(stitched, expected_height, chunks)
    return stitched
