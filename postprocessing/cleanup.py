"""Post-inference pixel cleanup.

Clamps near-black pixels to pure black to eliminate model hallucination patchiness
on solid regions: pixels where all channels are below the threshold are zeroed out.
Migrated from legacy/engine.py (_post_sharpen).
"""

import numpy as np


def cleanup_near_black(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    mask = np.all(image < threshold, axis=2)
    result = image.copy()
    result[mask] = 0
    return result
