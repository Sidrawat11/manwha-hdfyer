"""Experiment 10: 60-page confirmation of the detail detector (exp 8/9) on the real network.

Every page in the shared 60-page sample, two runs back to back:
  full      today's pipeline
  detail16  edges only, line OR detail detector (threshold 0.02), 16 px margin
Same pass bar as exp 6: time <= 80% of full, line regions >= 0.995, no page below 0.990.

Memory hardening vs exp 9:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (less fragmentation across crop shapes)
  - at most 4 crops per forward pass; on out-of-memory: clear cache, halve the batch, retry
  - GPU cache cleared and Python garbage collected after every page
  - a page whose time exceeds 3x the running median is flagged "spill" in the results
Recommended on Windows: NVIDIA Control Panel > CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback,
so a spill becomes a catchable out-of-memory error instead of a silent slowdown.

GPU: python experiments/exp10_detail_confirm.py      (resumable: rerun to continue)
Stop cleanly between pages: create outputs/exp10_detail_confirm/STOP (deleted on next start).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc, json, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
from common import load_sample, read_page, classify_detail
import exp06_split_vs_full as e6

OUT = "outputs/exp10_detail_confirm"
MAX_CROPS = 4

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
    safe_vram = 0.8 * bmap["available_vram"]
    sync = torch.cuda.synchronize
    oom_retries = {"n": 0}

    def _fwd(crops):
        x = torch.cat([_preprocess(c) for c in crops])
        xp, ph, pw = _pad(x, e6.SCALE)
        with torch.no_grad():
            y = model(xp)
        y = _unpad(y, ph, pw, e6.SCALE)
        out = [_postprocess(t) for t in torch.split(y, 1, dim=0)]
        del x, xp, y
        return out

    def forward(crops):
        outs, b, size = [], 0, MAX_CROPS
        while b < len(crops):
            try:
                outs += _fwd(crops[b:b + size]); b += size
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' not in str(e).lower(): raise
                torch.cuda.empty_cache(); oom_retries["n"] += 1
                if size == 1: raise
                size = max(1, size // 2)
        return outs

    def run_full(src):
        done = []
        for batch in build_batches(chunk_page(src, 0, bmap), bmap):
            done.extend(batch_upscale(batch, model, e6.SCALE))
        done.sort(key=lambda t: t[1].chunk_index)
        return reassemble(done, e6.SCALE)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.jsonl")
    stop_file = os.path.join(OUT, "STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    rows = []
    if os.path.exists(path):
        for l in open(path):
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass   # a line cut off by a hard kill: that page simply reruns
        with open(path, "w") as f:   # rewrite without any broken line
            f.writelines(json.dumps(r) + "\n" for r in rows)
    if rows:
        print(f"Resuming: {len(rows)} of 60 pages already done", flush=True)
    done_keys = {r["i"] for r in rows}
    log = open(path, "a")
    sample = load_sample()
    first = True
    for i, p in enumerate(sample):
        if i in done_keys: continue
        if os.path.exists(stop_file):
            print(f"STOP file found: stopping cleanly after {len(rows)} pages. Rerun to resume.", flush=True)
            break
        src = read_page(p["cbz"], p["page"])
        kinds = classify_detail(src, e6.TILE, 0.02)
        rec = {"i": i, "series": p["series"], "warm": not first,
               "squares_detail16": round(100 * float((kinds == 2).mean()), 1)}
        first = False
        oom_before = oom_retries["n"]
        sync(); t = time.perf_counter(); full4 = run_full(src); sync()
        rec["t_full"] = round(time.perf_counter() - t, 2)
        full2 = e6.half(full4); del full4
        sync(); t = time.perf_counter()
        r4, _, _ = e6.routed(src, kinds, 16, forward, safe_vram)
        sync(); rec["t_detail16"] = round(time.perf_counter() - t, 2)
        r2 = e6.half(r4); del r4
        rec.update({f"{k}_detail16": round(v, 4) for k, v in e6.region_scores(full2, r2, kinds).items()})
        rec["oom_retries"] = oom_retries["n"] - oom_before
        warm = [r for r in rows if r["warm"] and not r.get("spill")]
        med_f = st.median([r["t_full"] for r in warm]) if len(warm) >= 5 else None
        med_d = st.median([r["t_detail16"] for r in warm]) if len(warm) >= 5 else None
        rec["spill"] = bool(med_f and (rec["t_full"] > 3 * med_f or rec["t_detail16"] > 3 * med_d))
        if i < 6 or rec["ssim_detail16"] < 0.99:
            e6.save_visual(os.path.join(OUT, f"page{i:02d}_full_detail16_diff.png"), full2, r2, kinds)
        rows.append(rec)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f'{len(rows):2d}/60 page {i:2d} {p["series"][:20]:20s} full {rec["t_full"]:5.1f}s | '
              f'detail16 {rec["t_detail16"]:5.1f}s ssim {rec["ssim_detail16"]:.4f}'
              f'{"  SPILL" if rec["spill"] else ""}{"  oom-retry x%d" % rec["oom_retries"] if rec["oom_retries"] else ""}',
              flush=True)
        del full2, r2, src; gc.collect(); torch.cuda.empty_cache()
    log.close()
    if rows:
        summarize(rows)

def summarize(rows):
    timed = [r for r in rows if r["warm"] and not r["spill"]]
    tf = max(1e-9, sum(r["t_full"] for r in timed)); td = sum(r["t_detail16"] for r in timed)
    lines = [f"{len(rows)} pages for quality, {len(timed)} for timing ({sum(r['spill'] for r in rows)} spill pages excluded)",
             f"Time: detail16 = {100*td/tf:.0f}% of full   [bar: <= 80%]",
             f"Line regions similarity: mean {st.mean(r['ssim_edge_detail16'] for r in rows):.4f}   [bar: >= 0.995]",
             f"Worst page similarity: {min(r['ssim_detail16'] for r in rows):.4f}, "
             f"pages below 0.990: {sum(r['ssim_detail16'] < 0.99 for r in rows)}   [bar: none]",
             f"Squares to network: {st.mean(r['squares_detail16'] for r in rows):.0f}%"]
    for s in sorted({r["series"] for r in rows}):
        a = [r for r in rows if r["series"] == s]; t = [r for r in timed if r["series"] == s]
        lines.append(f"  {s}: time {100*sum(r['t_detail16'] for r in t)/max(1e-9, sum(r['t_full'] for r in t)):.0f}%, "
                     f"mean ssim {st.mean(r['ssim_detail16'] for r in a):.4f}, below 0.990: {sum(r['ssim_detail16'] < 0.99 for r in a)}")
    text = "\n".join(lines); print("\n" + text)
    open(os.path.join(OUT, "summary.txt"), "w").write(text + "\n")

if __name__ == "__main__":
    main()
