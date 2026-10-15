"""CPU finishing of upscaled pages.

Takes each stitched 4x page through sharpen -> black cleanup -> Lanczos half ->
encode and writes it to the CBZ. A page from a native 2x model is already at its
final size: it is sharpened with half the radius and not halved. By default the work runs on background threads
so it overlaps with the GPU (OpenCV releases the GIL in blur, resize and encode),
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


SHARPEN_RADIUS = {4: 1.0, 2: 0.5}   # the same blur width on the final 2x page either way


def finish_page(page: np.ndarray, packager: Packager, scale: int = 4) -> bytes:
    """
    Sharpens and cleans up a stitched page and returns it at 2x, encoded in the packager's format.
    A 4x page is halved after sharpening; a 2x page (scale=2) is left at its size.
    """
    if scale not in SHARPEN_RADIUS:
        raise ValueError(f"finish_page handles 4x and 2x pages, got {scale}x")
    page = sharpen(page, radius=SHARPEN_RADIUS[scale])
    page = cleanup_near_black(page)
    if scale == 4:
        h, w = page.shape[:2]
        page = cv2.resize(page, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)
    return packager.encode(page)


class PageFinisher:
    """Finishes pages and writes them to the packager in submission order. Supports use as a context manager."""

    def __init__(self, packager: Packager, workers: int = 4, max_inflight: int = 3, overlap: bool = True, scale: int = 4):
        if workers < 1 or max_inflight < 1:
            raise ValueError("workers and max_inflight must be at least 1")
        if scale not in SHARPEN_RADIUS:
            raise ValueError(f"PageFinisher handles 4x and 2x pages, got {scale}x")
        self.packager = packager
        self.scale = scale
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
            self.packager.add_encoded(name, finish_page(page, self.packager, self.scale))
            return

        while self._pending and self._pending[0][1].done():
            self._write_oldest()
        while len(self._pending) >= self.max_inflight:
            self._write_oldest()
        self._pending.append((name, self._executor.submit(finish_page, page, self.packager, self.scale)))

    def drain(self) -> None:
        """Waits for every page still in flight and writes them in order."""
        while self._pending:
            self._write_oldest()
