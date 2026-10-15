"""Batches: same-shape strips grouped for one forward pass each."""

from hdfyer.digest.strips import ChunkMeta
from collections import defaultdict
from typing import Callable


def build_batches(chunks: list[ChunkMeta], batch_size: int | Callable[[int, int], int]) -> list[list[ChunkMeta]]:
    """
    Groups same-shape chunks into batches of `batch_size` (a number, or a function of a chunk's height and width).
    Chunks are grouped across pages, so a batch can hold strips of several pages.
    Returns a list of batches, where each batch is a list of ChunkMetas that can be torch.stacked together.
    """
    slice_map = defaultdict(list)
    batches = []

    for chunk in chunks:
        h, w = chunk.page_slice.shape[:2]
        slice_map[(h, w)].append(chunk)
    
    ## Run through the dict keys and make batches
    for k, v in slice_map.items():
        h, w = k
        size = batch_size(h, w) if callable(batch_size) else batch_size

        for i in range(0, len(v), size):
            batches.append(v[i:i+size])

    return batches
