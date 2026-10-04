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

def reassemble(chunks: list[tuple[np.ndarray, ChunkMeta]], scale: int = 4) -> np.ndarray:
    """Stitches upscaled chunks into a full page with overlap blending. Receives paired upscaled arrays and their ChunkMeta, returns the reconstructed page."""
    chunk_body, chunk_meta = chunks[0]
    # Unchunked page: overlap_bottom is 0, and [:-0] would slice to an empty array.
    if len(chunks) == 1:
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
    
    return np.concatenate(pieces, axis=0)
