"""Packages processed images into a CBZ archive in JPEG or PNG format."""

import numpy as np
import cv2
import zipfile
from pathlib import PurePosixPath

# Every entry gets this timestamp (the earliest a zip can hold), so identical input gives a byte-identical CBZ.
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


class Packager:
    """Encodes and writes images into a CBZ archive. Supports use as a context manager."""

    def __init__(self, zip_file_name, fmt, quality):
        self.zip_file = zipfile.ZipFile(zip_file_name, 'w')
        self.quality = quality
        if fmt == 'jpg':
            self.format = '.jpg'
            self.fmt = cv2.IMWRITE_JPEG_QUALITY
        elif fmt == 'png':
            self.format = '.png'
            self.fmt = cv2.IMWRITE_PNG_COMPRESSION
            # PNG compression: quality > 90 treated as lossless
            self.quality = 0 if quality > 90 else 9
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'jpg' or 'png'.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def encode(self, img: np.ndarray) -> bytes:
        """Encode an image array in the archive's format. Touches no shared state, so it is safe to call from worker threads."""
        success, buffer = cv2.imencode(self.format, img, [self.fmt, self.quality])
        if not success:
            raise RuntimeError(f"Could not encode image as {self.format}")
        return buffer.tobytes()

    def add_encoded(self, filename, encoded: bytes):
        """Add already-encoded image bytes (from encode) to the archive, renaming the extension to match the encoded format."""
        self._write(str(PurePosixPath(filename).with_suffix(self.format)), encoded)

    def add_image(self, filename, img: np.ndarray):
        """Encode and add an image array to the archive, renaming the extension to match the encoded format."""
        self.add_encoded(filename, self.encode(img))

    def add_raw(self, filename, raw_bytes):
        """Write raw bytes directly into the archive without re-encoding."""
        self._write(filename, raw_bytes)

    def _write(self, filename: str, data: bytes):
        """Write one entry with the fixed timestamp and the same settings writestr gives a plain name."""
        info = zipfile.ZipInfo(filename, date_time=FIXED_DATE_TIME)
        info.compress_type = self.zip_file.compression
        info.external_attr = 0o600 << 16
        self.zip_file.writestr(info, data)

    def close(self):
        self.zip_file.close()
