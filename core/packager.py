"""Packages processed images into a CBZ archive in JPEG or PNG format."""

import numpy as np
import cv2
import zipfile
from pathlib import PurePosixPath


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

    def add_image(self, filename, img: np.ndarray):
        """Encode and add an image array to the archive, renaming the extension to match the encoded format."""
        success, buffer = cv2.imencode(self.format, img, [self.fmt, self.quality])
        if success:
            self.zip_file.writestr(str(PurePosixPath(filename).with_suffix(self.format)), buffer.tobytes())

    def add_raw(self, filename, raw_bytes):
        """Write raw bytes directly into the archive without re-encoding."""
        self.zip_file.writestr(filename, raw_bytes)

    def close(self):
        self.zip_file.close()
