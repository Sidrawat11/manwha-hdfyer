"""CPU finishing of upscaled pages.

Takes each stitched 2x page through sharpen -> black cleanup -> encode and writes it
to the CBZ. By default the work runs on background threads
so it overlaps with the GPU (OpenCV releases the GIL in blur and encode),
while the archive is only ever written by the calling thread, in the order the
pages were handed over.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from hdfyer.render.package import Packager


def sharpen(image: np.ndarray, strength: float = 0.3, radius: float = 1.0) -> np.ndarray:
    """Unsharp mask: GaussianBlur, then cv2.addWeighted to amplify the detail upscaling softened."""
    if strength == 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def cleanup_near_black(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Pixels whose channels are all below the threshold become pure black: removes patchiness in solid dark regions."""
    mask = np.all(image < threshold, axis=2)
    result = image.copy()
    result[mask] = 0
    return result


# On the final 2x page: the same blur width as the old pipeline's radius 1.0 at 4x, before it halved the page.
SHARPEN_RADIUS = 0.5


def finish_page(page: np.ndarray, packager: Packager) -> bytes:
    """Sharpens and cleans up a stitched 2x page and returns it encoded in the packager's format."""
    page = sharpen(page, radius=SHARPEN_RADIUS)
    page = cleanup_near_black(page)
    return packager.encode(page)


class PageFinisher:
    """Finishes pages and writes them to the packager in submission order. Supports use as a context manager."""

    def __init__(self, packager: Packager, workers: int = 4, max_inflight: int = 3, overlap: bool = True):
        if workers < 1 or max_inflight < 1:
            raise ValueError("workers and max_inflight must be at least 1")
        self.packager = packager
        self.max_inflight = max_inflight
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
