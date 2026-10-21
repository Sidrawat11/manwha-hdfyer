"""Exp 21, step 1: the student reference set, the student's counterpart of data/teacher/ultrasharp-2x/.

Every page the teacher set covers, through production's digest stage exactly (hdfyer-2x-s-r2, 512-row strips with
32-row overlap, the cached TensorRT FP16 engine and batch profile, linear-blend stitch), stopping before the render
stage: no sharpen, no near-black cleanup. Lossless PNG (compression 1), same layout as the teacher set:

    data/student/hdfyer-2x-s-r2-raw/training-v3/<label>/<chapter>/<page>.png
    data/student/hdfyer-2x-s-r2-raw/exams/<label>/<page>.png
    data/student/hdfyer-2x-s-r2-raw/manifest.json

Resumable: pages already written are skipped.

    python experiments/exp21/make_student.py
"""
import os, sys, time, zipfile

import cv2
import numpy as np
from tqdm import tqdm

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "cloud"))
os.chdir(ROOT)

from pathlib import Path  # noqa: E402
from hdfyer import pipeline  # noqa: E402
from hdfyer.digest import trt as trt_backend  # noqa: E402
from hdfyer.digest.batches import build_batches  # noqa: E402
from hdfyer.digest.engine import prepare_chapter, upscale_stream  # noqa: E402
from hdfyer.digest.stitch import PageBuffer  # noqa: E402
from hdfyer.digest.strips import chunk_page  # noqa: E402
from outputs import output_dir  # noqa: E402
import upscale_teacher as ut  # noqa: E402  (job list and layout shared with the teacher set)

TEACHER = "data/teacher/ultrasharp-2x"
OUT = "data/student/hdfyer-2x-s-r2-raw"
GROUP = 16  # pages chunked and batched together (bounds memory; batches still mix pages of one width)


def main():
    jobs = [j for j in ut.load_jobs() if os.path.exists(os.path.join(TEACHER, j["out"]))]
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, j["out"]))]
    print(f"{len(jobs)} pages in the teacher set, {len(jobs) - len(todo)} already done, {len(todo)} to do -> {OUT}/")
    weights = pipeline.WEIGHTS
    model, backend, batch = pipeline.load_runtime(weights, "auto", "auto")
    print(f"backend {backend}")
    t0 = time.time()
    by_src = {}
    for j in todo:
        by_src.setdefault(j["src"], []).append(j)
    bar = tqdm(total=len(todo), unit="page", desc="student raw")
    for src, js in by_src.items():
        with zipfile.ZipFile(ut.local_src(src)) as z:
            for g in range(0, len(js), GROUP):
                group = js[g:g + GROUP]
                chunks, sizes = [], {}
                for pid, j in enumerate(group):
                    img = cv2.imdecode(np.frombuffer(z.read(j["entry"]), np.uint8), cv2.IMREAD_COLOR)  # as hdfyer.ingest.reader
                    sizes[pid] = img.shape[:2]
                    for c in chunk_page(img, pid, pipeline.STRIP_HEIGHT):
                        c.page_name = j["out"]
                        chunks.append(c)
                batches = build_batches(chunks, batch)
                prepare_chapter(chunks, pipeline.SCALE)
                buf = PageBuffer(pipeline.SCALE)
                for _, results in upscale_stream(batches, model, pipeline.SCALE):
                    for up, meta in results:
                        page = buf.add(up, meta)
                        if page is None:
                            continue
                        j = group[meta.page_id]
                        h, w = sizes[meta.page_id]
                        assert page.shape[:2] == (2 * h, 2 * w), f"student output size on {j['out']}"
                        dst = os.path.join(OUT, j["out"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
                        cv2.imwrite(dst + ".tmp.png", page, [cv2.IMWRITE_PNG_COMPRESSION, 1]); os.replace(dst + ".tmp.png", dst)
                        bar.update(1)
                assert buf.is_empty(), "a page was left half-stitched"
    bar.close()
    done = [j for j in jobs if os.path.exists(os.path.join(OUT, j["out"]))]
    import teacher_jobs as tj
    output_dir(OUT, model={"file": str(weights), "sha256": trt_backend.file_sha256(weights)},
               settings={"backend": backend, "precision": "fp16", "scale": pipeline.SCALE, "strip_height": pipeline.STRIP_HEIGHT,
                         "overlap": 32, "stitch": "linear blend (hdfyer.digest.stitch)", "round_output": True,
                         "finishing": "none (no sharpen, no near-black cleanup)", "png_compression": 1},
               layout="same as data/teacher/ultrasharp-2x", page_counts=tj.page_counts(done),
               seconds_this_run=round(time.time() - t0, 1))
    print(f"done: {len(done)}/{len(jobs)} pages in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
