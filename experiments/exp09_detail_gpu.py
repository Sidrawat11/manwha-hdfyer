"""Experiment 9: does the detail detector (exp 8) fix the failing pages on the real network?

For 10 pages (the 4 that failed in exp 6 + 2 passing controls per series), runs back to back:
  full     today's pipeline
  line16   edges only, old line detector, 16 px margin   (exp 6 rule, re-timed in the same process)
  detail16 edges only, line OR detail detector (threshold 0.02), 16 px margin
Memory fix vs exp 6: GPU cache cleared after every page; batches capped at 8 crops / 60% VRAM.

GPU: python experiments/exp09_detail_gpu.py      (resumable)
"""
import gc, json, os, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
from common import load_sample, read_page, line_mask, classify_tiles, classify_detail
import exp06_split_vs_full as e6

PAGES = [14, 18, 32, 35, 3, 9, 23, 26, 45, 52]
FAIL = {14, 18, 32, 35}
OUT = "experiments/out/exp09_detail_gpu"

def main():
    import torch
    import torch.backends.cudnn as cudnn
    from core.model_loader import load_rrdbnet
    from profiler.cache import load_cache
    from inference.chunker import chunk_page
    from inference.batcher import build_batches
    from inference.engine import batch_upscale, _preprocess, _pad, _unpad, _postprocess
    from pipeline.reassembler import reassemble
    cudnn.benchmark = True
    model = load_rrdbnet("models/4x-UltraSharp.pth")
    bmap = load_cache()
    safe_vram = 0.8 * bmap["available_vram"]       # routed() budgets 75% of this = 60% of the card
    sync = torch.cuda.synchronize

    def forward(crops):
        outs = []
        for b in range(0, len(crops), 8):          # hard cap: 8 crops per forward pass
            x = torch.cat([_preprocess(c) for c in crops[b:b + 8]])
            xp, ph, pw = _pad(x, e6.SCALE)
            with torch.no_grad():
                y = model(xp)
            y = _unpad(y, ph, pw, e6.SCALE)
            outs += [_postprocess(t) for t in torch.split(y, 1, dim=0)]
            del x, xp, y
        return outs

    def run_full(src):
        done = []
        for batch in build_batches(chunk_page(src, 0, bmap), bmap):
            done.extend(batch_upscale(batch, model, e6.SCALE))
        done.sort(key=lambda t: t[1].chunk_index)
        return reassemble(done, e6.SCALE)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.jsonl")
    done_keys = set()
    if os.path.exists(path):
        done_keys = {json.loads(l)["i"] for l in open(path) if l.strip()}
    log = open(path, "a")
    sample = load_sample()
    first = True
    for n, i in enumerate(PAGES):
        if i in done_keys: continue
        p = sample[i]
        src = read_page(p["cbz"], p["page"]); h, w = src.shape[:2]
        rules = {"line16": classify_tiles(src, line_mask(src), e6.TILE),
                 "detail16": classify_detail(src, e6.TILE, 0.02)}
        rec = {"i": i, "series": p["series"], "failed_in_exp6": i in FAIL, "warm": not first}
        first = False
        sync(); t = time.perf_counter(); full4 = run_full(src); sync()
        rec["t_full"] = round(time.perf_counter() - t, 2)
        full2 = e6.half(full4); del full4
        for name, kinds in rules.items():
            rec[f"squares_{name}"] = round(100 * float((kinds == 2).mean()), 1)
            sync(); t = time.perf_counter()
            r4, _, _ = e6.routed(src, kinds, 16, forward, safe_vram)
            sync(); rec[f"t_{name}"] = round(time.perf_counter() - t, 2)
            r2 = e6.half(r4); del r4
            rec.update({f"{k}_{name}": round(v, 4) for k, v in e6.region_scores(full2, r2, kinds).items()})
            if name == "detail16":
                e6.save_visual(os.path.join(OUT, f"page{i:02d}_full_detail16_diff.png"), full2, r2, kinds)
            del r2
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f'{n+1:2d}/{len(PAGES)} page {i:2d} {"FAILED-before" if i in FAIL else "control      "} '
              f'full {rec["t_full"]:5.1f}s | line16 {rec["t_line16"]:5.1f}s ssim {rec["ssim_line16"]:.4f} | '
              f'detail16 {rec["t_detail16"]:5.1f}s ssim {rec["ssim_detail16"]:.4f}', flush=True)
        del full2, src; gc.collect(); torch.cuda.empty_cache()
    log.close()

    rows = [json.loads(l) for l in open(path) if l.strip()]
    warm = [r for r in rows if r["warm"]] or rows
    tf = sum(r["t_full"] for r in warm)
    lines = [f"{len(rows)} pages ({len(warm)} timed)"]
    for name in ("line16", "detail16"):
        key_t, key_s = "t_" + name, "ssim_" + name
        failed = ", ".join("%d:%.4f" % (r["i"], r[key_s]) for r in rows if r["failed_in_exp6"])
        lines.append("%s: time %.0f%% of full | ssim all %.4f | failed-before pages %s | below 0.990: %d" % (
            name, 100 * sum(r[key_t] for r in warm) / tf, st.mean(r[key_s] for r in rows),
            failed, sum(r[key_s] < 0.99 for r in rows)))
    text = "\n".join(lines); print("\n" + text)
    open(os.path.join(OUT, "summary.txt"), "w").write(text + "\n")

if __name__ == "__main__":
    main()
