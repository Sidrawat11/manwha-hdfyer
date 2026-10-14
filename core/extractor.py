"""
Image extraction module for the ManwhaHDFyer pipeline.

Handles reading CBZ archive files and extracting their contents as
decoded image arrays for downstream processing.
"""

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
import cv2
import imagesize
import numpy as np

def extract_images(cbz_path):
    with zipfile.ZipFile(cbz_path) as z:
        file_list = _get_image_names(z)
        for name in file_list:
            raw_bytes = z.read(name)
            image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield name, image, raw_bytes

def load_chapter(cbz_path, threads=4):
    """
    Every page extract_images would yield, as a list of (name, image), decoded on `threads` threads.
    The archive is read on the calling thread (a ZipFile can't be shared between threads); OpenCV's decoders
    release the GIL, so the decodes run in parallel. Same images as extract_images, same order.
    """
    with zipfile.ZipFile(cbz_path) as z:
        raw = [(name, z.read(name)) for name in _get_image_names(z)]
    with ThreadPoolExecutor(threads, thread_name_prefix="decode") as pool:
        images = list(pool.map(lambda item: cv2.imdecode(np.frombuffer(item[1], np.uint8), cv2.IMREAD_COLOR), raw))
    return [(name, image) for (name, _), image in zip(raw, images) if image is not None]

def page_sizes(cbz_path):
    """(width, height) of every page extract_images would yield, read from the image headers without decoding."""
    sizes = []
    with zipfile.ZipFile(cbz_path) as z:
        for name in _get_image_names(z):
            with z.open(name) as raw:
                sizes.append(imagesize.get(io.BytesIO(raw.read(65536))))
    return sizes

def count_images(cbz_path):
    with zipfile.ZipFile(cbz_path) as z:
        return len(_get_image_names(z))

def _get_image_names(z):
    include_filter = ('.jpg', '.png', '.jpeg', '.webp')
    exclude_filter = ('__MACOSX', '.')
    return sorted([f for f in z.namelist() if f.endswith(include_filter) and not f.startswith(exclude_filter)])