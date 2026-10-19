"""Experiment 12: strip-height sweep on the full pipeline, on 15 FRESH random pages.

Experiment 11 found small crops run cheaper per pixel than 898-row strips. Is today's strip simply
too tall for this card? And is today's 128-row overlap overkill (routing showed 16 px a side hides seams)?

Settings (strip height for an 800 px-wide page / overlap; heights scale with width like the real chunker):
  today  898 / 128   (exactly today's chunker: fixed overlap, short last strip, so a new tail
                      shape per page height -> cuDNN tuning cost, as in the real pipeline)
  A      898 / 32    (isolates the overlap saving)
  B      512 / 32
  C      256 / 32
  D      128 / 32
Strips have one fixed height per setting; starts are spread evenly so every overlap is about the
same and no strip is swallowed by its neighbours' overlaps.
Batch size per setting = as many strips as fit in 60% of VRAM (max 8); out-of-memory halves it.
Quality of each setting is scored against "today" at 2x (raw network output, no sharpening).

Pages: 5 per series from chapters NOT used in the 60-page sample (seed 2027).
Showcase, per page, in outputs/exp12_strip_sweep/showcase/:
  pageNN_full_2x.jpg        the finished 2x page exactly as the pipeline makes it (sharpen + cleanup)
  pageNN_closeup.jpg        most detailed 900-row window: reader-app enlargement | upscaler

GPU: python experiments/exp12_strip_sweep.py     (resumable; STOP file in the out folder stops it cleanly)
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc, glob, json, random, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import cv2, numpy as np
from common import page_names, read_page, load_sample, detail_scores
import exp06_split_vs_full as e6

OUT = "outputs/exp12_strip_sweep"
SHOW = os.path.join(OUT, "showcase")
SETTINGS = [("today", 898, 128), ("A", 898, 32), ("B", 512, 32), ("C", 256, 32), ("D", 128, 32)]
SCALE = 4

def fresh_pages(per_series=5, seed=2027):
    used = {p["cbz"] for p in load_sample()}
    rng = random.Random(seed); pages = []
    for series in sorted(os.listdir("data/library")):
        cbzs = [c for c in sorted(glob.glob(os.path.join("data/library", series, "*.cbz"))) if c not in used]
        for cbz in rng.sample(cbzs, per_series):
            pages.append({"series": series, "cbz": cbz, "page": rng.choice(page_names(cbz))})
    return pages

def strip_plan(h, H, ov):
    """[(y0, y1)] strips of fixed height H, starts spread evenly down the page.
    Every overlap is >= ov (roughly ov), and no strip's top + bottom overlap can exceed H:
    the reassembler silently duplicates or drops rows when that happens (found here, exp 12)."""
    if h <= H: return [(0, h)]
    n = -(-(h - H) // (H - ov)) + 1
    def build(n):
        return [(round(k * (h - H) / (n - 1)), round(k * (h - H) / (n - 1)) + H) for k in range(n)]
    plan = build(n)
    def ok(plan):
        return all((plan[k][0] - plan[k - 1][0] if False else 0) == 0 and
                   ((plan[k - 1][1] - plan[k][0]) if k else 0) + ((plan[k][1] - plan[k + 1][0]) if k < len(plan) - 1 else 0) <= H
                   for k in range(len(plan)))
    while not ok(plan) and n > 2:
        n -= 1; plan = build(n)
    return plan

def today_plan(h, H=898, ov=128):
    """Exactly what inference/chunker.chunk_page does today: fixed 128 overlap, short last strip."""
    if h <= H: return [(0, h)]
    plan, y = [], 0
    while True:
        if y + H >= h:
            plan.append((y, h)); break
        plan.append((y, y + H)); y += H - ov
    return plan

def main():
    import torch
    import torch.backends.cudnn as cudnn
    from core.model_loader import load_rrdbnet
    from profiler.cache import load_cache
    from inference.chunker import ChunkMeta
    from inference.engine import _preprocess, _pad, _unpad, _postprocess
    from pipeline.reassembler import reassemble
    from postprocessing.sharpen import sharpen
    from postprocessing.cleanup import cleanup_near_black
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

    def run(src, H, ov, faithful=False):
        h, w = src.shape[:2]
        # Heights are defined for 800 px-wide pages; scale to keep the same pixels per strip, exactly as
        # the real chunker does (get_max_chunk_height). A fixed 898 rows on a 1080 px page needs 8.3 GB.
        H = max(64, int(H * 800 / w))
        plan = today_plan(h, H, ov) if faithful else strip_plan(h, H, ov)
        groups = {}                                   # batch only same-shape strips together
        for k, (a, b) in enumerate(plan):
            groups.setdefault(b - a, []).append(k)
        ups = [None] * len(plan)
        size = max(1, min(8, int((0.6 * vram - 32.1) / (0.0085 * (plan[0][1] - plan[0][0]) * w))))
        for hh, idx in groups.items():
            bsz, b = max(1, min(8, int((0.6 * vram - 32.1) / (0.0085 * hh * w)))), 0
            while b < len(idx):
                try:
                    res = fwd([src[plan[k][0]:plan[k][1]] for k in idx[b:b + bsz]])
                    for k, r in zip(idx[b:b + bsz], res): ups[k] = r
                    b += bsz
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if 'out of memory' not in str(e).lower(): raise
                    torch.cuda.empty_cache()
                    if bsz == 1: raise
                    bsz //= 2
        n = len(plan); metas = []
        for k, (a, bb) in enumerate(plan):
            top = plan[k - 1][1] - a if k > 0 else 0
            bot = bb - plan[k + 1][0] if k < n - 1 else 0
            metas.append(ChunkMeta(0, None, k, n, a, bb, top, bot))
        out = reassemble(list(zip(ups, metas)), SCALE)
        assert out.shape[0] == h * SCALE, f"stitch produced {out.shape[0]} rows, expected {h * SCALE}"
        return out, len(plan), size

    os.makedirs(SHOW, exist_ok=True)
    pages = fresh_pages()
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
        rec = {"i": i, "series": p["series"], "chapter": os.path.basename(p["cbz"]), "page": p["page"],
               "size": f"{w}x{h}", "warm": not first}
        if first:   # warm-up pass for every setting so cuDNN tuning isn't timed
            for nm, H, ov in SETTINGS: run(src, H, ov, faithful=(nm == "today"))
            first = False; rec["warm"] = True
        ref2 = None
        for name, H, ov in SETTINGS:
            sync(); t = time.perf_counter(); out4, n, bs = run(src, H, ov, faithful=(name == "today")); sync()
            rec[f"t_{name}"] = round(time.perf_counter() - t, 2)
            rec[f"strips_{name}"] = n; rec[f"batch_{name}"] = bs
            out2 = e6.half(out4)
            if name == "today":
                ref2 = out2
                final = e6.half(cleanup_near_black(sharpen(out4)))          # exactly what the pipeline ships
                cv2.imwrite(os.path.join(SHOW, f"page{i:02d}_full_2x.jpg"), final, [cv2.IMWRITE_JPEG_QUALITY, 92])
                sc = detail_scores(src, 128).mean(axis=1)                    # most detailed 900-row window (input rows)
                rows_per = 900 // 2 // 128 + 1
                best = int(np.argmax(np.convolve(sc, np.ones(rows_per), "valid"))) if len(sc) > rows_per else 0
                y0 = min(best * 128, max(0, h - 450)); y1 = min(h, y0 + 450)
                app = cv2.resize(src[y0:y1], None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(SHOW, f"page{i:02d}_closeup.jpg"),
                            np.hstack([app, np.full((app.shape[0], 16, 3), 255, np.uint8), final[2*y0:2*y1]]),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                del final
            else:
                s = e6.ssim_map(ref2, out2)
                rec[f"ssim_{name}"] = round(float(s.mean()), 4)
                rec[f"ssim_min_row_{name}"] = round(float(s.mean(axis=1).min()), 4)   # worst row: seams show here
            del out4, out2
        log.write(json.dumps(rec) + "\n"); log.flush(); rows.append(rec)
        print(f'{len(rows):2d}/{len(pages)} page {i:2d} {p["series"][:20]:20s} ' +
              " | ".join(f'{n} {rec["t_"+n]:5.1f}s' + (f' {rec["ssim_"+n]:.4f}' if n != "today" else "")
                         for n, _, _ in SETTINGS), flush=True)
        del ref2, src; gc.collect(); torch.cuda.empty_cache()
    log.close()
    timed = [r for r in rows if r["warm"]]
    if not timed: return
    base = sum(r["t_today"] for r in timed)
    lines = [f"{len(rows)} fresh pages ({len(timed)} timed)"]
    for name, H, ov in SETTINGS:
        tt = sum(r[f"t_{name}"] for r in timed)
        q = "" if name == "today" else (f" | similarity to today {st.mean(r['ssim_'+name] for r in rows):.4f}, "
                                        f"worst row {min(r['ssim_min_row_'+name] for r in rows):.4f}")
        lines.append(f"{name:5s} {H:4d}/{ov:3d}: {100*tt/base:5.0f}% of today's time{q}")
    text = "\n".join(lines); print("\n" + text)
    open(os.path.join(OUT, "summary.txt"), "w").write(text + "\n")

if __name__ == "__main__":
    main()
