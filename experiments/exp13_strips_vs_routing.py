"""Experiment 13: smarter strips vs edge routing, head to head, on 15 NEW random pages.

Runs per page, same process, same timer:
  today   exactly today's chunker (898 rows for 800 px, 128 overlap, short last strip)
  strips  smarter strips, exp 12 setting B: 512 rows for 800 px (scaled by width), 32 overlap,
          fixed shape, evenly spaced
  route   edge routing, exp 10: detail detector (0.02), 128 px squares, 16 px margin
Quality: each scored against "today" at 2x. Pass bar as before: no page below 0.990.
Pages: 5 per series from chapters used in neither the 60-page sample nor exp 12 (seed 2028).

Seam check (end of run): exp 12 pages 4 and 14, where one band of rows scored low at 32-row overlap.
Saves out/exp13_strips_vs_routing/seam_pageNN.jpg: [today | strips | difference x8] around the worst rows.

GPU: python experiments/exp13_strips_vs_routing.py   (resumable; STOP file in the out folder stops it)
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc, glob, json, random, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import cv2, numpy as np
from common import page_names, read_page, load_sample, classify_detail
from exp12_strip_sweep import strip_plan, today_plan
import exp06_split_vs_full as e6

OUT = "experiments/out/exp13_strips_vs_routing"
SCALE = 4
SEAM_PAGES = [4, 14]   # indices into exp 12's pages.json

def new_pages(per_series=5, seed=2028):
    used = {p["cbz"] for p in load_sample()}
    used |= {p["cbz"] for p in json.load(open("experiments/out/exp12_strip_sweep/pages.json"))}
    rng = random.Random(seed); pages = []
    for series in sorted(os.listdir("Manwhas")):
        cbzs = [c for c in sorted(glob.glob(os.path.join("Manwhas", series, "*.cbz"))) if c not in used]
        for cbz in rng.sample(cbzs, per_series):
            pages.append({"series": series, "cbz": cbz, "page": rng.choice(page_names(cbz))})
    return pages

def is_oom(e):
    return "out of memory" in str(e).lower()

def main():
    import torch
    import torch.backends.cudnn as cudnn
    from core.model_loader import load_rrdbnet
    from profiler.cache import load_cache
    from inference.chunker import ChunkMeta
    from inference.engine import _preprocess, _pad, _unpad, _postprocess
    from pipeline.reassembler import reassemble
    cudnn.benchmark = True
    model = load_rrdbnet("models/4x-UltraSharp.pth")
    vram = load_cache()["available_vram"]
    sync = torch.cuda.synchronize

    def fwd(imgs):
        x = torch.cat([_preprocess(c) for c in imgs]); xp, ph, pw = _pad(x, SCALE)
        with torch.no_grad(): y = model(xp)
        y = _unpad(y, ph, pw, SCALE)
        out = [_postprocess(t) for t in torch.split(y, 1, dim=0)]; del x, xp, y
        return out

    def forward_batched(imgs, bsz):
        """Same-shape images, batched; on out-of-memory halve the batch and retry."""
        outs, b = [], 0
        while b < len(imgs):
            try:
                outs += fwd(imgs[b:b + bsz]); b += bsz
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not is_oom(e) or bsz == 1: raise
                torch.cuda.empty_cache(); bsz //= 2
        return outs

    def run_strips(src, H, ov, faithful):
        h, w = src.shape[:2]
        H = max(64, int(H * 800 / w))
        plan = today_plan(h, H, ov) if faithful else strip_plan(h, H, ov)
        groups = {}
        for k, (a, b) in enumerate(plan): groups.setdefault(b - a, []).append(k)
        ups = [None] * len(plan)
        for hh, idx in groups.items():
            bsz = max(1, min(8, int((0.6 * vram - 32.1) / (0.0085 * hh * w))))
            for k, r in zip(idx, forward_batched([src[plan[k][0]:plan[k][1]] for k in idx], bsz)): ups[k] = r
        n = len(plan)
        metas = [ChunkMeta(0, None, k, n, a, b, plan[k-1][1] - a if k else 0, b - plan[k+1][0] if k < n-1 else 0)
                 for k, (a, b) in enumerate(plan)]
        out = reassemble(list(zip(ups, metas)), SCALE)
        assert out.shape[0] == h * SCALE, f"stitch produced {out.shape[0]} rows, expected {h * SCALE}"
        return out

    def run_route(src):
        kinds = classify_detail(src, e6.TILE, 0.02)
        r4, _, _ = e6.routed(src, kinds, 16, lambda crops: forward_batched(crops, 4), 0.8 * vram)
        return r4, round(100 * float((kinds == 2).mean()), 1)

    os.makedirs(OUT, exist_ok=True)
    pages = new_pages()
    json.dump(pages, open(os.path.join(OUT, "pages.json"), "w"), indent=1)
    path = os.path.join(OUT, "results.jsonl"); stop = os.path.join(OUT, "STOP")
    if os.path.exists(stop): os.remove(stop)
    rows = []
    if os.path.exists(path):
        for l in open(path):
            try: rows.append(json.loads(l))
            except ValueError: pass
        open(path, "w").writelines(json.dumps(r) + "\n" for r in rows)
    done = {r["i"] for r in rows}
    log = open(path, "a"); first = True
    for i, p in enumerate(pages):
        if i in done: continue
        if os.path.exists(stop):
            print("STOP file found: stopping cleanly. Rerun to resume.", flush=True); break
        src = read_page(p["cbz"], p["page"]); h, w = src.shape[:2]
        if first:   # warm-up so cuDNN tuning isn't timed
            run_strips(src, 898, 128, True); run_strips(src, 512, 32, False); run_route(src)
            first = False
        rec = {"i": i, "series": p["series"], "chapter": os.path.basename(p["cbz"]), "page": p["page"], "size": f"{w}x{h}"}
        sync(); t = time.perf_counter(); o = run_strips(src, 898, 128, True); sync()
        rec["t_today"] = round(time.perf_counter() - t, 2); ref2 = e6.half(o); del o
        sync(); t = time.perf_counter(); o = run_strips(src, 512, 32, False); sync()
        rec["t_strips"] = round(time.perf_counter() - t, 2); s2 = e6.half(o); del o
        sync(); t = time.perf_counter(); o, sq = run_route(src); sync()
        rec["t_route"] = round(time.perf_counter() - t, 2); r2 = e6.half(o); del o
        rec["squares_route"] = sq
        for name, img in (("strips", s2), ("route", r2)):
            m = e6.ssim_map(ref2, img)
            rec[f"ssim_{name}"] = round(float(m.mean()), 4)
            rec[f"ssim_min_row_{name}"] = round(float(m.mean(axis=1).min()), 4)
        log.write(json.dumps(rec) + "\n"); log.flush(); rows.append(rec)
        print(f'{len(rows):2d}/{len(pages)} page {i:2d} {p["series"][:20]:20s} today {rec["t_today"]:5.1f}s | '
              f'strips {rec["t_strips"]:5.1f}s {rec["ssim_strips"]:.4f} | route {rec["t_route"]:5.1f}s {rec["ssim_route"]:.4f}',
              flush=True)
        del ref2, s2, r2, src; gc.collect(); torch.cuda.empty_cache()
    log.close()

    # ---- seam check on exp 12's flagged pages
    exp12 = json.load(open("experiments/out/exp12_strip_sweep/pages.json"))
    for k in SEAM_PAGES:
        p = exp12[k]; src = read_page(p["cbz"], p["page"])
        ref2 = e6.half(run_strips(src, 898, 128, True)); s2 = e6.half(run_strips(src, 512, 32, False))
        rows_ssim = e6.ssim_map(ref2, s2).mean(axis=1); y = int(np.argmin(rows_ssim))
        y0, y1 = max(0, y - 150), min(ref2.shape[0], y + 150)
        diff = cv2.convertScaleAbs(cv2.absdiff(ref2[y0:y1], s2[y0:y1]), alpha=8)
        gap = np.full((y1 - y0, 12, 3), 255, np.uint8)
        cv2.imwrite(os.path.join(OUT, f"seam_page{k:02d}.jpg"),
                    np.hstack([ref2[y0:y1], gap, s2[y0:y1], gap, diff]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"seam check exp12 page {k}: worst row {y} (2x), row similarity {rows_ssim[y]:.4f} -> seam_page{k:02d}.jpg", flush=True)
        del ref2, s2, src; gc.collect(); torch.cuda.empty_cache()

    if not rows: return
    base = sum(r["t_today"] for r in rows)
    lines = [f"{len(rows)} new pages",
             f"today : 100%",
             f"strips: {100*sum(r['t_strips'] for r in rows)/base:.0f}% | mean similarity {st.mean(r['ssim_strips'] for r in rows):.4f} | "
             f"worst page {min(r['ssim_strips'] for r in rows):.4f} | faster than today on {sum(r['t_strips'] < r['t_today'] for r in rows)}/{len(rows)}",
             f"route : {100*sum(r['t_route'] for r in rows)/base:.0f}% | mean similarity {st.mean(r['ssim_route'] for r in rows):.4f} | "
             f"worst page {min(r['ssim_route'] for r in rows):.4f} | faster than strips on {sum(r['t_route'] < r['t_strips'] for r in rows)}/{len(rows)}",
             f"route vs strips: {100*sum(r['t_route'] for r in rows)/sum(r['t_strips'] for r in rows):.0f}%"]
    text = "\n".join(lines); print("\n" + text)
    open(os.path.join(OUT, "summary.txt"), "w").write(text + "\n")

if __name__ == "__main__":
    main()
