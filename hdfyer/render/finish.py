"""CPU finishing of upscaled pages.

Takes each stitched 2x page through sharpen (edge-masked USM) -> black cleanup -> encode and writes it
to the CBZ. By default the work runs on background threads
so it overlaps with the GPU (OpenCV releases the GIL in blur and encode),
while the archive is only ever written by the calling thread, in the order the
pages were handed over.
"""

import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from hdfyer.render.package import Packager


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with radius ceil(3 sigma) and reflect-101 borders (the kernel experiments/exp21/sharpen.py uses)."""
    r = max(1, math.ceil(3 * sigma))
    return cv2.GaussianBlur(x, (2 * r + 1, 2 * r + 1), sigma, borderType=cv2.BORDER_REFLECT_101)


LUMA = (0.114, 0.587, 0.299)  # B, G, R


def sharpen(image: np.ndarray, amount: float = 1.0, radius: float = 0.8, lo: float = 6.0, hi: float = 30.0) -> np.ndarray:
    """
    Edge-masked unsharp mask (Exp 21's winner): the luma detail (luma minus a gaussian of `radius`) is added to all
    three channels, weighted by a mask that is 0 in flat areas and gradients and 1 on edges. The mask is a smoothstep
    of the edge strength (Sobel gradient of the luma blurred by 0.7, levels per pixel) between lo and hi, blurred by
    1.0. No noise gain in flat areas, no colour fringes. Matches experiments/exp21/sharpen.usm_edge within 1 level.
    """
    if amount == 0:
        return image
    b, g, r = cv2.split(image)
    y = cv2.addWeighted(b, LUMA[0], g, LUMA[1], 0, dtype=cv2.CV_32F)
    y = cv2.scaleAdd(r.astype(np.float32), LUMA[2], y)
    detail = cv2.subtract(y, _gaussian(y, radius))
    yb = _gaussian(y, 0.7)
    t = cv2.magnitude(cv2.Sobel(yb, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE),
                      cv2.Sobel(yb, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE))
    t *= 0.125 / (hi - lo)          # Sobel / 8 = levels per pixel; then (strength - lo) / (hi - lo)
    t -= lo / (hi - lo)
    np.clip(t, 0, 1, out=t)
    mask = t * t                    # smoothstep t*t*(3 - 2t), in place
    t *= -2
    t += 3
    mask *= t
    delta = _gaussian(mask, 1.0)
    delta *= detail
    if amount != 1.0:
        delta *= amount
    delta = np.rint(delta, out=delta).astype(np.int16)
    return cv2.add(image, cv2.merge((delta, delta, delta)), dtype=cv2.CV_8U)   # saturating


def cleanup_near_black(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Pixels whose channels are all below the threshold become pure black: removes patchiness in solid dark regions.
    (inRange + a masked copy into zeros: the same bytes as np.all(image < threshold, axis=2) and boolean indexing,
    about 8x faster on a full page.)"""
    keep = cv2.bitwise_not(cv2.inRange(image, (0, 0, 0), (threshold - 1,) * 3))
    result = np.zeros_like(image)
    cv2.copyTo(image, keep, result)
    return result


# Exp 21: edge-masked USM at amount 1.0, radius 0.8 brings hdfyer's edges to UltraSharp's acutance (0.88 -> 1.03)
# with no flat-area noise gain. It replaced a plain unsharp mask (strength 0.3, radius 0.5) on all three channels.
SHARPEN_AMOUNT = 1.0
SHARPEN_RADIUS = 0.8

# OpenCV threads per call while pages are finished on the worker pool. Each worker's blur/Sobel/encode fanning out
# over every core on top of the pool oversubscribes the CPU: on Ch134 (RTX 4060 laptop, 20 cores) one thread per
# call finished the chapter 0.2-1.3 s faster than OpenCV's default, for the old and the new finish alike, in three
# comparisons with about 0.5 s of run-to-run noise (experiments/exp21/time_finish.py). None leaves OpenCV alone.
WORKER_OPENCV_THREADS = 1


def finish_page(page: np.ndarray, packager: Packager) -> bytes:
    """Sharpens and cleans up a stitched 2x page and returns it encoded in the packager's format."""
    page = sharpen(page, amount=SHARPEN_AMOUNT, radius=SHARPEN_RADIUS)
    page = cleanup_near_black(page)
    return packager.encode(page)


class PageFinisher:
    """Finishes pages and writes them to the packager in submission order. Supports use as a context manager."""

    def __init__(self, packager: Packager, workers: int = 4, max_inflight: int = 3, overlap: bool = True):
        if workers < 1 or max_inflight < 1:
            raise ValueError("workers and max_inflight must be at least 1")
        self.packager = packager
        self.max_inflight = max_inflight
        if overlap and WORKER_OPENCV_THREADS is not None:
            cv2.setNumThreads(WORKER_OPENCV_THREADS)   # process-wide; the worker pool is the parallelism
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="finish") if overlap else None
        self._pending = deque()   # (page name, future) in submission order

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.drain()
        finally:
            self._shutdown()
        return False

    def _shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _write_oldest(self) -> None:
        """Waits for the oldest page and writes it. A worker's exception is raised here, on the calling thread."""
        name, future = self._pending.popleft()
        self.packager.add_encoded(name, future.result())

    def submit(self, name: str, page: np.ndarray) -> None:
        """
        Hands over a stitched page. Without overlap it is finished and written before returning.
        With overlap it is queued for a worker; at most max_inflight pages are queued or in progress,
        and when that limit is reached this waits for the oldest one first.
        """
        if self._executor is None:
            self.packager.add_encoded(name, finish_page(page, self.packager))
            return

        while self._pending and self._pending[0][1].done():
            self._write_oldest()
        while len(self._pending) >= self.max_inflight:
            self._write_oldest()
        self._pending.append((name, self._executor.submit(finish_page, page, self.packager)))

    def drain(self) -> None:
        """Waits for every page still in flight and writes them in order."""
        while self._pending:
            self._write_oldest()
