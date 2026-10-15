"""Shared constants and stand-ins: a nearest-neighbour 'model', and fakes that raise the GPU's errors."""

import json
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from hdfyer.digest import trt as trt_backend
from hdfyer.ingest.reader import _get_image_names
from hdfyer.digest.stitch import reassemble
from hdfyer.digest.strips import chunk_page, crop_upscaled


SCALE = 4
SWEEP_WIDTHS = [690, 800, 1080]
SWEEP_HEIGHTS = range(130, 9001, 7)
SHORT_HEIGHTS = [1, 2, 3, 31, 32, 33, 64, 127, 255, 256, 257, 400, 510, 511, 512, 513, 543, 544, 545, 991, 992, 993, 1023, 1024, 1025]
SWEEP_COLUMNS = 8
STUDENT_SCALE = 2
STUDENT_STRIP_HEIGHT = 512


def fake_upscale(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def round_trip(image: np.ndarray, upscale=fake_upscale, scale: int = SCALE, strip_height: int = 512) -> tuple[int, int]:
    """Returns (chunk count, max pixel difference vs. the expected page)."""
    chunks = chunk_page(image, 0, strip_height)
    heights = {c.page_slice.shape[0] for c in chunks}
    assert heights == {strip_height}, f"fixed-shape strips must all be {strip_height} rows, got {sorted(heights)}"
    stitched = reassemble([(crop_upscaled(upscale(c.page_slice), c, scale), c) for c in chunks], scale)
    expected = upscale(image)
    assert stitched.shape[0] == image.shape[0] * scale, f"{stitched.shape[0]} rows != {image.shape[0] * scale}"
    assert stitched.shape == expected.shape, f"shape {stitched.shape} != {expected.shape}"
    return len(chunks), int(np.abs(stitched.astype(int) - expected.astype(int)).max())


def fake_upscale_2x(image: np.ndarray) -> np.ndarray:
    return fake_upscale(image, STUDENT_SCALE)


def fake_upscale_column(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """fake_upscale on the first few columns only. Sweep pages are constant along x, so nothing is lost."""
    return fake_upscale(np.ascontiguousarray(image[:, :SWEEP_COLUMNS]), scale)


def fake_upscale_column_2x(image: np.ndarray) -> np.ndarray:
    return fake_upscale_column(image, STUDENT_SCALE)


def sweep_page(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """A w x h page of random rows, constant along x: chunking and stitching only act on rows, and this keeps the sweep fast."""
    rows = rng.integers(0, 256, (h, 1, 3), dtype=np.uint8)
    return np.broadcast_to(rows, (h, w, 3))


class FakeOutOfMemory:
    """A 2x nearest-neighbour stand-in for the model that raises out-of-memory the way the GPU does."""

    def __init__(self, largest_batch: int = 99, message: str = "CUDA out of memory. Tried to allocate 4.39 GiB", fail_first: int = 0):
        self.largest_batch, self.message, self.fail_first = largest_batch, message, fail_first
        self.calls = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(x.shape[0])
        if x.shape[0] > self.largest_batch or len(self.calls) <= self.fail_first:
            raise RuntimeError(self.message)
        return torch.nn.functional.interpolate(x, scale_factor=STUDENT_SCALE, mode="nearest")


class FakeTrt:
    """Stands in for trt_backend.TrtBackend: engines of a fixed batch that can fail to build or run."""

    def __init__(self, run_model, largest_build: int = 99, build_message: str = "CUDA driver error: out of memory"):
        self.run_model, self.largest_build, self.build_message = run_model, largest_build, build_message
        self.built, self.released = [], 0
        self.switches, self.load_seconds, self.build_seconds = 0, [], []

    def has_engine(self, height, width, batch=1):
        return (height, width, batch) in self.built

    def build(self, height, width, batch=1):
        if batch > self.largest_build:
            raise trt_backend.EngineBuildError(f"engine build for {height}x{width} (batch {batch}) failed ({self.build_message})")
        self.built.append((height, width, batch))

    def release(self):
        self.released += 1

    def run(self, x):
        batch, _, height, width = x.shape
        if not self.has_engine(height, width, batch):
            self.build(height, width, batch)
        return self.run_model(x)


def extract_images(cbz_path):
    """Reference decoder: every page load_chapter returns, read and decoded one at a time. Yields (name, image, bytes)."""
    with zipfile.ZipFile(cbz_path) as z:
        for name in _get_image_names(z):
            raw_bytes = z.read(name)
            image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield name, image, raw_bytes
