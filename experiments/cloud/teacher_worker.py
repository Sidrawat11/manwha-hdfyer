"""Runs on the rented GPU: 4x-UltraSharp -> Lanczos half for every job in jobs.json, exactly as distill.py's teacher stage.

Same load_model (spandrel, fp16 via .half()), run_model (512-row strips, 32 overlap, linear blend), half() and PNG
settings, imported from distill.py. Optional strip batching (a page's strips in one forward pass, up to 8 at a time)
is used only if its saved 2x output is within 1 level of the one-strip-at-a-time path on the first --check-pages
pages. Outputs already present are skipped; each PNG is written to a temp name and renamed. Progress goes to
progress.json for the laptop to poll.
"""
import argparse, json, os, sys, time, traceback, zipfile
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill as d  # copied next to this file on the server


BATCH = {"max": 8}  # strips per forward pass; halved for good on out-of-memory


def run_batched(desc, img):
    import torch
    h = img.shape[0]
    plan = d.plan_strips(h, min(d.STRIP, h))  # every strip of a page has the same shape
    ups = []
    with torch.inference_mode():
        i = 0
        while i < len(plan):
            chunk = plan[i:i + BATCH["max"]]
            try:
                y = desc(torch.cat([d.to_t(img[a:b]) for a, b in chunk]))
            except RuntimeError as e:
                if "out of memory" not in str(e) or BATCH["max"] == 1:
                    raise
                y = None; torch.cuda.empty_cache(); BATCH["max"] //= 2
                continue
            ups += [d.from_t(y[k:k + 1]) for k in range(len(chunk))]
            i += len(chunk)
    return d.stitch(ups, plan, desc.scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="jobs.json")
    ap.add_argument("--model", default="models/4x-UltraSharp.pth")
    ap.add_argument("--out", default="out")
    ap.add_argument("--batch", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--check-pages", type=int, default=5)
    ap.add_argument("--max-batch", type=int, default=8)
    a = ap.parse_args()
    BATCH["max"] = a.max_batch
    import torch
    torch.backends.cudnn.benchmark = True  # as distill.py's teacher stage
    jobs = json.load(open(a.jobs))
    prog = {"phase": "loading model", "total": len(jobs), "done": 0, "skipped": 0, "started": time.time(),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch": None, "times": {}}

    def save():
        prog["updated"] = time.time()
        json.dump(prog, open("progress.json.tmp", "w")); os.replace("progress.json.tmp", "progress.json")

    save()
    try:
        teacher = d.load_model(a.model)
        assert teacher.scale == 4, "teacher should be the 4x model"
        todo = []
        for j in jobs:
            if os.path.exists(os.path.join(a.out, j["out"])):
                prog["skipped"] += 1
            else:
                todo.append(j)
        prog["done"] = prog["skipped"]
        batched = a.batch == "on"
        by_src = {}
        for j in todo:
            by_src.setdefault(j["src"], []).append(j)
        warm, checked = False, 0
        prog["phase"] = "upscaling"; save()
        for src, js in by_src.items():
            with zipfile.ZipFile(src) as z:
                for j in js:
                    img = cv2.imdecode(np.frombuffer(z.read(j["entry"]), np.uint8), cv2.IMREAD_COLOR)
                    if not warm:
                        d.run_model(teacher, img[:256]); warm = True  # warm-up, as distill.py
                    d.sync(); t = time.perf_counter()
                    if a.batch == "auto" and checked < a.check_pages:
                        prog["phase"] = "checking batching"; save()
                        one = d.run_model(teacher, img)
                        many = run_batched(teacher, img)
                        # compared on what gets saved: the 2x output after half()
                        diff = int(np.abs(d.half(one).astype(np.int16) - d.half(many).astype(np.int16)).max())
                        prev = prog["batch"] or {"pages": 0, "max_diff": 0}
                        prog["batch"] = {"pages": prev["pages"] + 1, "max_diff": max(prev["max_diff"], diff)}
                        checked += 1
                        if checked == a.check_pages:
                            batched = prog["batch"]["max_diff"] <= 1
                            prog["batch"]["used"] = batched
                        up = one  # the reference path's output is what gets saved during the check
                        prog["phase"] = "upscaling"
                    else:
                        up = run_batched(teacher, img) if batched else d.run_model(teacher, img)
                    out2 = d.half(up)
                    prog["max_batch"] = BATCH["max"]
                    d.sync(); prog["times"][j["out"]] = round(time.perf_counter() - t, 3)
                    assert out2.shape[:2] == (img.shape[0] * 2, img.shape[1] * 2), f"teacher output size on {j['out']}"
                    dst = os.path.join(a.out, j["out"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
                    cv2.imwrite(dst + ".tmp.png", out2, [cv2.IMWRITE_PNG_COMPRESSION, 1]); os.replace(dst + ".tmp.png", dst)
                    prog["done"] += 1; save()
        if a.batch == "auto" and prog["batch"] and "used" not in prog["batch"]:
            prog["batch"]["used"] = False  # fewer pages than the check needs: stayed on the reference path
        prog["phase"] = "finished"; save()
    except BaseException:
        prog["phase"] = "error"; prog["error"] = traceback.format_exc()[-3000:]; save()
        raise


if __name__ == "__main__":
    main()
