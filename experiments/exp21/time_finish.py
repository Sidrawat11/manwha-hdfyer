"""Exp 21 follow-up: what shipping edge-masked USM costs on Ch134 (the 34-page, 800 px bench chapter).

1. Finish cost per page: Ch134's stitched 2x pages (production's digest, in memory) through each finish, one page at
   a time on one thread, 3 passes, median per page:
     old              plain USM (0.3, r0.5) + the old numpy near-black cleanup + JPEG 92   (production before)
     new              edge-masked USM (1.0, r0.8) + the OpenCV cleanup + JPEG 92           (production now)
     new USM, old cleanup                                                                    (for attribution)
2. Chapter wall time: hdfyer.pipeline.process_chapter_v2 on Ch134 (4 finishing threads overlapping the GPU, as
   production runs), interleaved, 3 rounds each: production before (old finish, OpenCV's default thread count),
   the new finish at OpenCV's default, and production now (new finish, one OpenCV thread per finishing worker).

    python experiments/exp21/time_finish.py   -> experiments/exp21/finish_timing.csv
"""
import csv, glob, os, statistics, sys, tempfile, time
from pathlib import Path

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from hdfyer import pipeline  # noqa: E402
from hdfyer.render import finish  # noqa: E402
from hdfyer.render.package import Packager  # noqa: E402
from hdfyer.digest.batches import build_batches  # noqa: E402
from hdfyer.digest.engine import prepare_chapter, upscale_stream  # noqa: E402
from hdfyer.digest.stitch import PageBuffer  # noqa: E402
from hdfyer.digest.strips import chunk_page  # noqa: E402
from hdfyer.ingest import reader  # noqa: E402

CHAPTER = Path(sorted(glob.glob("data/bench/ch134/*/*.cbz"))[0])
CSV = os.path.join(HERE, "finish_timing.csv")


def old_sharpen(image, strength=0.3, radius=0.5):
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return np.clip(cv2.addWeighted(image, 1 + strength, blurred, -strength, 0), 0, 255).astype(np.uint8)


def old_cleanup(image, threshold=15):
    mask = np.all(image < threshold, axis=2)
    result = image.copy()
    result[mask] = 0
    return result


FINISHES = {
    "old": lambda page, pk: pk.encode(old_cleanup(old_sharpen(page))),
    "new": finish.finish_page,
    "new USM, old cleanup": lambda page, pk: pk.encode(old_cleanup(finish.sharpen(page))),
}


def stitched_pages(model, batch):
    pages = [img for _, img in reader.load_chapter(CHAPTER)]
    chunks = [c for pid, img in enumerate(pages) for c in chunk_page(img, pid, pipeline.STRIP_HEIGHT)]
    prepare_chapter(chunks, pipeline.SCALE)
    buf, out = PageBuffer(pipeline.SCALE), {}
    for _, results in upscale_stream(build_batches(chunks, batch), model, pipeline.SCALE):
        for up, meta in results:
            page = buf.add(up, meta)
            if page is not None:
                out[meta.page_id] = page
    return [out[k] for k in sorted(out)]


def main():
    model, backend, batch = pipeline.load_runtime(pipeline.WEIGHTS, "auto", "auto")
    pages = stitched_pages(model, batch)
    mpx = statistics.fmean(p.shape[0] * p.shape[1] for p in pages) / 1e6
    print(f"Ch134: {len(pages)} stitched pages, {mpx:.1f} Mpx each on average; backend {backend}")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        pk = Packager(Path(tmp) / "x.cbz", "jpg", 92)
        cv2.setNumThreads(1)  # one finishing worker's view
        for name, fn in FINISHES.items():
            fn(pages[0], pk)
            per_page = []
            for p in pages:
                ts = []
                for _ in range(3):
                    t = time.perf_counter(); fn(p, pk); ts.append(time.perf_counter() - t)
                per_page.append(statistics.median(ts) * 1000)
            ms = statistics.fmean(per_page)
            rows.append({"measure": "finish per page, 1 thread", "variant": name, "ms": round(ms, 1), "ms_per_mpx": round(ms / mpx, 2)})
            print(f"finish {name:22s} {ms:7.1f} ms/page ({ms / mpx:.1f} ms/Mpx)")
        real, real_threads = finish.finish_page, finish.WORKER_OPENCV_THREADS
        configs = {"before: old finish, OpenCV default threads": ("old", None),
                   "new finish, OpenCV default threads": ("new", None),
                   "now: new finish, 1 OpenCV thread per worker": ("new", 1)}
        walls = {k: [] for k in configs}
        for rnd in range(3):
            for label, (name, threads) in configs.items():
                cv2.setNumThreads(-1)  # OpenCV's default; PageFinisher lowers it when WORKER_OPENCV_THREADS is set
                finish.finish_page, finish.WORKER_OPENCV_THREADS = FINISHES[name], threads
                t = time.perf_counter()
                pipeline.process_chapter_v2(CHAPTER, model, Path(tmp) / "x.cbz", batch_size=batch, quiet=True)
                walls[label].append(time.perf_counter() - t)
                print(f"round {rnd + 1}: {label:45s} {walls[label][-1]:.2f} s", flush=True)
        finish.finish_page, finish.WORKER_OPENCV_THREADS = real, real_threads
        for label, ws in walls.items():
            rows.append({"measure": "Ch134 chapter wall time, ms (median of 3)", "variant": label,
                         "ms": round(statistics.median(ws) * 1000), "ms_per_mpx": ""})
            print(f"{label:45s} median {statistics.median(ws):.2f} s")
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, ["measure", "variant", "ms", "ms_per_mpx"]); w.writeheader(); w.writerows(rows)
    print(f"wrote {CSV}")


if __name__ == "__main__":
    main()
