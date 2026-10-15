"""render: finishing pages (serial and overlapped) and writing reproducible CBZs."""

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
from hdfyer.render import finish as finisher_module
from hdfyer.render.finish import PageFinisher, cleanup_near_black, sharpen
from hdfyer.render.package import Packager
from tests.helpers import *


def write_cbz(path: Path, pages: list[tuple[str, np.ndarray]], overlap: bool, workers: int = 4, max_inflight: int = 3, scale: int = SCALE) -> int:
    """Runs fake stitched pages through the finishing path into a CBZ. Returns the most pages that were in flight at once."""
    most_inflight = 0
    with Packager(path, 'jpg', 92) as packager, PageFinisher(packager, workers, max_inflight, overlap, scale) as page_finisher:
        for name, page in pages:
            page_finisher.submit(name, page)
            most_inflight = max(most_inflight, len(page_finisher._pending))
    return most_inflight


def cbz_entries(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as z:
        return [(name, z.read(name)) for name in z.namelist()]


def checks() -> list[tuple[str, bool]]:
    """(description, passed) for each page-finishing check. No GPU: the pages are random arrays."""
    checks = []
    rng = np.random.default_rng(2)
    # Big pages first, so with several workers the later pages finish before the earlier ones.
    heights = [2400, 1800, 96, 64, 1200, 32, 900, 48, 600, 16, 300, 8]
    pages = [(f"{i:03d}__page.png", rng.integers(0, 256, (h, 640, 3), dtype=np.uint8)) for i, h in enumerate(heights)]
    pages[3][1][:] = 5   # a near-black page, for the cleanup step
    expected_names = [str(Path(name).with_suffix(".jpg")) for name, _ in pages]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_cbz(tmp / "serial.cbz", pages, overlap=False)
        serial = cbz_entries(tmp / "serial.cbz")
        checks.append(("serial: every page written, in order", [name for name, _ in serial] == expected_names))
        decoded = cv2.imdecode(np.frombuffer(serial[0][1], np.uint8), cv2.IMREAD_COLOR)
        checks.append(("serial: pages are halved JPEGs", decoded is not None and decoded.shape == (1200, 320, 3)))
        black = cv2.imdecode(np.frombuffer(serial[3][1], np.uint8), cv2.IMREAD_COLOR)
        checks.append(("serial: near-black page is cleaned to black", black is not None and int(black.max()) == 0))

        for workers, max_inflight in [(4, 3), (1, 1), (8, 2), (2, 6)]:
            label = f"overlap, {workers} workers, {max_inflight} in flight"
            most = write_cbz(tmp / "overlap.cbz", pages, overlap=True, workers=workers, max_inflight=max_inflight)
            overlapped = cbz_entries(tmp / "overlap.cbz")
            checks.append((f"{label}: same page order as serial", [n for n, _ in overlapped] == [n for n, _ in serial]))
            checks.append((f"{label}: every page byte-identical to serial", overlapped == serial))
            checks.append((f"{label}: never more than {max_inflight} pages in flight", 1 <= most <= max_inflight))

        # 2x pages (native 2x model): kept at their size, sharpened with radius 0.5, cleaned, same in both paths.
        write_cbz(tmp / "serial2x.cbz", pages, overlap=False, scale=STUDENT_SCALE)
        write_cbz(tmp / "overlap2x.cbz", pages, overlap=True, scale=STUDENT_SCALE)
        serial2x = cbz_entries(tmp / "serial2x.cbz")
        checks.append(("2x pages: overlapped CBZ byte-identical to serial",
                       (tmp / "serial2x.cbz").read_bytes() == (tmp / "overlap2x.cbz").read_bytes()))
        decoded = cv2.imdecode(np.frombuffer(serial2x[0][1], np.uint8), cv2.IMREAD_COLOR)
        checks.append(("2x pages: not halved", decoded is not None and decoded.shape == (2400, 640, 3)))
        expected = cleanup_near_black(sharpen(pages[4][1], radius=0.5))
        ok, encoded = cv2.imencode(".jpg", expected, [cv2.IMWRITE_JPEG_QUALITY, 92])
        checks.append(("2x pages: sharpen radius 0.5 then black cleanup, nothing else", ok and encoded.tobytes() == serial2x[4][1]))
        black = cv2.imdecode(np.frombuffer(serial2x[3][1], np.uint8), cv2.IMREAD_COLOR)
        checks.append(("2x pages: near-black page is cleaned to black", black is not None and int(black.max()) == 0))
        try:
            write_cbz(tmp / "bad.cbz", pages[:1], overlap=False, scale=3)
            rejected = False
        except ValueError:
            rejected = True
        checks.append(("a scale other than 2 or 4 is rejected", rejected))

        # Pages really are finished off the calling thread, and several at once.
        real_finish = finisher_module.finish_page
        threads, running, most_running, lock = set(), 0, 0, threading.Lock()

        def tracked_finish(page, packager, scale=SCALE):
            nonlocal running, most_running
            with lock:
                threads.add(threading.current_thread().name)
                running += 1
                most_running = max(most_running, running)
            time.sleep(0.05)
            try:
                return real_finish(page, packager, scale)
            finally:
                with lock:
                    running -= 1

        def failing_finish(page, packager, scale=SCALE):
            if page.shape[0] == 96:
                raise RuntimeError("simulated finishing failure")
            return real_finish(page, packager, scale)

        try:
            finisher_module.finish_page = tracked_finish
            write_cbz(tmp / "tracked.cbz", pages, overlap=True, workers=4, max_inflight=3)
            checks.append(("overlap: pages are finished on worker threads",
                           threads and threading.current_thread().name not in threads))
            checks.append(("overlap: several pages are finished at once, within the limit", 2 <= most_running <= 3))
            threads.clear()
            write_cbz(tmp / "tracked.cbz", pages, overlap=False)
            checks.append(("serial: pages are finished on the calling thread", threads == {threading.current_thread().name}))

            # Same pages packaged twice, more than a zip timestamp tick (2 s) apart: the files must be byte-identical.
            finisher_module.finish_page = real_finish
            write_cbz(tmp / "first.cbz", pages, overlap=True)
            time.sleep(2.1)
            write_cbz(tmp / "second.cbz", pages, overlap=True)
            checks.append(("packaging the same pages twice gives byte-identical CBZ files",
                           (tmp / "first.cbz").read_bytes() == (tmp / "second.cbz").read_bytes()))
            checks.append(("serial and overlapped CBZ files are byte-identical",
                           (tmp / "serial.cbz").read_bytes() == (tmp / "second.cbz").read_bytes()))
            with zipfile.ZipFile(tmp / "first.cbz") as z:
                checks.append(("every zip entry has the fixed 1980-01-01 timestamp",
                               all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in z.infolist())))
                checks.append(("zip entries keep their compression and permissions",
                               all(info.compress_type == zipfile.ZIP_STORED and info.external_attr == 0o600 << 16
                                   for info in z.infolist())))

            # A failing page must fail the whole run, in both paths, and nothing after it may be written.
            finisher_module.finish_page = failing_finish
            for overlap in (True, False):
                label = "overlap" if overlap else "serial"
                try:
                    write_cbz(tmp / "failed.cbz", pages, overlap=overlap)
                    raised = False
                except RuntimeError as e:
                    raised = "simulated finishing failure" in str(e)
                checks.append((f"{label}: a worker exception fails the run", raised))
                written = [name for name, _ in cbz_entries(tmp / "failed.cbz")]
                checks.append((f"{label}: pages after the failed one are not written", written == expected_names[:2]))
        finally:
            finisher_module.finish_page = real_finish

    return checks
