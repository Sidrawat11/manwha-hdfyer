"""Exp 19 step 3: what does INT8 change? Every exam page (A-E, 150 pages) upscaled with the FP16 engine (production's
own, the reference) and with each INT8 variant; raw 2x output, before sharpening, compared per page.

Pages go through production's own code: chunk_page -> engine._preprocess -> TensorRT -> engine._to_bgr_uint8 (rounded)
-> crop_upscaled -> reassemble. Only the engine file differs.

Per page, INT8 vs FP16:
  psnr_db              over all pixels and channels
  max_abs_diff         largest difference in any channel
  pct_pixels_gt2       share of pixels where some channel differs by more than 2 levels
  flat_shift_b/g/r     mean signed difference per channel over flat regions (FP16 local std < 1.5 in a 9x9 window): colour wobble
  smooth_new_steps_per_mpx   in smooth regions (FP16 local std < 4), pixels where INT8 has a step of >= 2 levels to its
                       right or lower neighbour and FP16 has < 1: new banding edges, per megapixel of smooth area
  smooth_grad_ratio    mean |gradient| in smooth regions, INT8 / FP16 (1.0 = equally smooth)

  PYTHONPATH=experiments/out/exp19/pylib python experiments/exp19/quality.py
    -> experiments/exp19/quality_pages.csv, quality_by_exam.csv; comparers in experiments/out/exp19/
"""
import csv, glob, json, os, sys, zipfile
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import cv2, numpy as np, torch, tensorrt as trt
from tqdm import tqdm
from hdfyer.digest import engine as prod_engine
from hdfyer.digest.strips import chunk_page, crop_upscaled
from hdfyer.digest.stitch import reassemble
sys.path.insert(0, os.path.join(os.getcwd(), "experiments", "exp19"))
import engines

STRIP, SCALE = 512, 2
PRECISIONS = ["fp16", "int8-default", "int8-mixed"]
OUT = "experiments/out/exp19"


class Runner:
    """One loaded engine per (precision, width); runs a padded 1x3xHxW fp16 tensor."""
    def __init__(self):
        self.runtime = trt.Runtime(engines.LOGGER); self.live = {}; self.stream = torch.cuda.Stream()

    def get(self, precision, h, w):
        key = (precision, h, w)
        if key not in self.live:
            for k in list(self.live):   # one engine per precision at a time keeps VRAM low
                if k[0] == precision: del self.live[k]
            path = engines.build_fp16(h, w) if precision == "fp16" else engines.build_int8(precision, h, w)[0]
            eng = self.runtime.deserialize_cuda_engine(open(path, "rb").read())
            self.live[key] = (eng, eng.create_execution_context())
        return self.live[key]

    def run(self, precision, x):
        eng, ctx = self.get(precision, x.shape[2], x.shape[3])
        out_name = eng.get_tensor_name(1)
        y = torch.empty(tuple(ctx.get_tensor_shape(out_name)), device="cuda",
                        dtype=torch.float16 if eng.get_tensor_dtype(out_name) == trt.DataType.HALF else torch.float32)
        x = x.half().contiguous() if eng.get_tensor_dtype(eng.get_tensor_name(0)) == trt.DataType.HALF else x.float().contiguous()
        ctx.set_tensor_address(eng.get_tensor_name(0), x.data_ptr()); ctx.set_tensor_address(out_name, y.data_ptr())
        main = torch.cuda.current_stream(); self.stream.wait_stream(main)
        assert ctx.execute_async_v3(self.stream.cuda_stream)
        main.wait_stream(self.stream)
        return y.half()


def upscale_page(runner, precision, img):
    done = []
    for chunk in chunk_page(img, 0, STRIP):
        x, ph, pw = prod_engine._pad(prod_engine._preprocess(chunk.page_slice), SCALE)
        y = prod_engine._unpad(runner.run(precision, x), ph, pw, SCALE)
        done.append((crop_upscaled(prod_engine._to_bgr_uint8(y, True).contiguous().cpu().numpy(), chunk, SCALE), chunk))
    return reassemble(done, SCALE)


def local_std(gray, k=9):
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k)); m2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def compare(ref, test):
    d = test.astype(np.int16) - ref.astype(np.int16); a = np.abs(d)
    mse = float((d.astype(np.float64) ** 2).mean())
    gray_r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY); gray_t = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    std = local_std(gray_r)
    flat, smooth = std < 1.5, std < 4.0
    shift = [float(d[..., c][flat].mean()) if flat.any() else 0.0 for c in range(3)]
    def steps(g):
        g = g.astype(np.int16)
        gx = np.zeros_like(g); gy = np.zeros_like(g)
        gx[:, :-1] = np.abs(g[:, 1:] - g[:, :-1]); gy[:-1] = np.abs(g[1:] - g[:-1])
        return np.maximum(gx, gy)
    sr, st = steps(gray_r), steps(gray_t)
    new_steps = int(((st >= 2) & (sr < 1) & smooth).sum())
    smooth_px = int(smooth.sum())
    return {"psnr_db": round(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse), 2), "max_abs_diff": int(a.max()),
            "pct_pixels_gt2": round(100 * float((a.max(axis=2) > 2).mean()), 3),
            "flat_shift_b": round(shift[0], 3), "flat_shift_g": round(shift[1], 3), "flat_shift_r": round(shift[2], 3),
            "flat_pct": round(100 * float(flat.mean()), 1),
            "smooth_new_steps_per_mpx": round(new_steps / max(smooth_px, 1) * 1e6, 1),
            "smooth_grad_ratio": round(float(st[smooth].mean()) / max(float(sr[smooth].mean()), 1e-6), 3) if smooth_px else 1.0}


def exam_pages():
    key = json.load(open("exams/series_key.json"))
    label_of = {e["series"]: k for k, e in key.items()}
    out = []
    for d in sorted(glob.glob("exams/*/")):
        label = label_of[json.load(open(os.path.join(d, "manifest.json")))["series"]]
        with zipfile.ZipFile(os.path.join(d, "test.cbz")) as z:
            for name in sorted(z.namelist()):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    out.append((label, name, z.read(name)))
    return sorted(out, key=lambda p: (p[0], p[1]))


def main():
    runner = Runner()
    pages = exam_pages()
    rows = []
    for label, name, data in tqdm(pages, desc="exam pages"):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        outs = {p: upscale_page(runner, p, img) for p in PRECISIONS}
        for p in PRECISIONS[1:]:
            rows.append({"exam": label, "page": os.path.splitext(name)[0], "variant": p,
                         "size_2x": f"{outs['fp16'].shape[1]}x{outs['fp16'].shape[0]}", **compare(outs["fp16"], outs[p])})
    with open("experiments/exp19/quality_pages.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary = []
    for p in PRECISIONS[1:]:
        for label in sorted({r["exam"] for r in rows}) + ["all"]:
            R = [r for r in rows if r["variant"] == p and (label == "all" or r["exam"] == label)]
            summary.append({"variant": p, "exam": label, "pages": len(R),
                            "psnr_mean_db": round(np.mean([r["psnr_db"] for r in R]), 2), "psnr_min_db": min(r["psnr_db"] for r in R),
                            "max_abs_diff": max(r["max_abs_diff"] for r in R),
                            "pct_pixels_gt2_mean": round(np.mean([r["pct_pixels_gt2"] for r in R]), 3),
                            "flat_shift_mean_abs": round(np.mean([np.mean([abs(r[f"flat_shift_{c}"]) for c in "bgr"]) for r in R]), 3),
                            "flat_shift_worst_abs": round(max(max(abs(r[f"flat_shift_{c}"]) for c in "bgr") for r in R), 3),
                            "smooth_new_steps_per_mpx_mean": round(np.mean([r["smooth_new_steps_per_mpx"] for r in R]), 1),
                            "smooth_grad_ratio_mean": round(np.mean([r["smooth_grad_ratio"] for r in R]), 3)})
    with open("experiments/exp19/quality_by_exam.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    for s in summary:
        print(s)

    # Comparers: the 5 worst pages by PSNR (either variant) and the worst page of each exam.
    worst = {}
    for r in sorted(rows, key=lambda r: r["psnr_db"]):
        worst.setdefault((r["exam"], r["page"]), r["psnr_db"])
    chosen = list(worst)[:5]
    for label in sorted({r["exam"] for r in rows}):
        page = min((r for r in rows if r["exam"] == label), key=lambda r: r["psnr_db"])["page"]
        if (label, page) not in chosen:
            chosen.append((label, page))
    import exp15_native2x as e15
    e15.MODE_NOTE = "Raw 2x output (before sharpening). Left of the slider: FP16, production's engine. Right: the INT8 variant picked."
    by_key = {(l, os.path.splitext(n)[0]): d for l, n, d in pages}
    for i, (label, page) in enumerate(tqdm(chosen, desc="comparers")):
        img = cv2.imdecode(np.frombuffer(by_key[(label, page)], np.uint8), cv2.IMREAD_COLOR)
        outs = {p: upscale_page(runner, p, img) for p in PRECISIONS}
        rec = {f"ssim_{p.replace('-', '_')}": next(r["psnr_db"] for r in rows if r["exam"] == label and r["page"] == page and r["variant"] == p)
               for p in PRECISIONS[1:]}
        e15.write_compare(f"{OUT}/compare_{i:02d}_exam{label}_page{page}_full.html", i, f"exam {label} &middot; page {page} &middot; PSNR vs FP16 in the table column 'Similarity'",
                          img, {"today": outs["fp16"], "int8_default": outs["int8-default"], "int8_mixed": outs["int8-mixed"]},
                          {"today": "FP16 (production)", "int8_default": "INT8 default", "int8_mixed": "INT8 mixed"},
                          {"t_today": None, **rec}, full=True)
    print("comparers:", chosen)


if __name__ == "__main__":
    main()
