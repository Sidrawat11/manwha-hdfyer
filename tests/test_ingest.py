"""ingest: reading and decoding chapters, the library's to-do list, decode-ahead."""

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


def checks() -> list[tuple[str, bool]]:
    """Reading pages, finding the chapters to do, and decoding the next chapter ahead."""
    from hdfyer.ingest import library, reader
    from hdfyer.ingest import reader as extractor
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cbz = tmp / "lib" / "series" / "ch1.cbz"
        cbz.parent.mkdir(parents=True)
        rng = np.random.default_rng(4)
        with zipfile.ZipFile(cbz, "w") as z:
            for name, (h, w) in [("002.jpg", (300, 690)), ("001.png", (40, 1080)), ("003.webp", (90, 800))]:
                ok, data = cv2.imencode(Path(name).suffix, rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
                z.writestr(name, data.tobytes())
            z.writestr("__MACOSX/._001.png", b"junk")
            z.writestr("ComicInfo.xml", b"<x/>")
        decoded = [(image.shape[1], image.shape[0]) for _, image, _ in extractor.extract_images(cbz)]
        checks.append(("parallel: the warm-up's header scan sees exactly the pages the workers decode",
                       extractor.page_sizes(cbz) == decoded == [(1080, 40), (690, 300), (800, 90)]))

        (tmp / "lib" / "series" / "ch2.cbz").write_bytes(cbz.read_bytes())
        (tmp / "lib" / "series" / "ch3.cbz").write_bytes(cbz.read_bytes())
        out = tmp / "out" / "series"
        out.mkdir(parents=True)
        (out / "ch1.cbz").write_bytes(b"finished")
        (out / "ch2.cbz.part").write_bytes(b"interrupted")
        todo = [str(relative) for _, relative in library.todo_chapters(tmp / "lib", tmp / "out")]
        checks.append(("parallel: finished chapters are skipped and an interrupted .part is redone",
                       todo == ["series/ch2.cbz", "series/ch3.cbz"]))

        decoded_in_threads = extractor.load_chapter(cbz, threads=3)
        checks.append(("decode ahead: threaded decoding gives the same pages, in the same order",
                       [n for n, _ in decoded_in_threads] == [n for n, _, _ in extractor.extract_images(cbz)]
                       and all(np.array_equal(a, b) for (_, a), (_, _, b) in
                               zip(decoded_in_threads, [(n, i, i) for n, i, _ in extractor.extract_images(cbz)]))))

    # Prefetch: chapters come back in order, the next one is loaded while the caller works, never more than one ahead.
    saved_load = reader.load_chapter
    loaded, in_flight, most = [], [0], [0]
    lock = threading.Lock()

    def slow_load(path, threads):
        with lock:
            in_flight[0] += 1
            most[0] = max(most[0], in_flight[0])
        time.sleep(0.05)
        loaded.append(path)
        with lock:
            in_flight[0] -= 1
        if path == "bad":
            raise ValueError("corrupt chapter")
        return [(f"{path}-page", path)]

    reader.load_chapter = slow_load
    try:
        order, ahead = [], []
        for (path, _), pages in reader.prefetched([(f"c{i}", None) for i in range(5)]):
            time.sleep(0.1)   # "upscaling": the next chapter should load meanwhile
            order.append(pages[0][1])
            ahead.append(len(loaded) - len(order))
        checks.append(("decode ahead: chapters arrive in order, each loaded while the previous one is processed",
                       order == ["c0", "c1", "c2", "c3", "c4"] and ahead == [1, 1, 1, 1, 0] and most[0] == 1))
        got = []
        try:
            for (path, _), pages in reader.prefetched([("c0", None), ("bad", None), ("c2", None)]):
                got.append(path)
            raised = False
        except ValueError:
            raised = True
        checks.append(("decode ahead: a chapter that fails to load raises when its turn comes, after the ones before it",
                       raised and got == ["c0"]))
    finally:
        reader.load_chapter = saved_load

    return checks
