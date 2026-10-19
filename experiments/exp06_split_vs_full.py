"""Split vs full: does running the network only on line-art regions match the full pipeline?

For every page in outputs/sample.json (60 pages, 3 series):
  FULL    today's pipeline core: chunk_page -> build_batches -> batch_upscale -> reassemble (4x)
  ROUTEDm bicubic 4x base, then network output pasted over every run of edge tiles
          (128px tiles, runs merged horizontally), with an m-pixel context margin (m = 16, 32)
Both are then Lanczos-halved to 2x (what the reader sees) and compared.
Sharpen/cleanup are skipped for both: identical post-steps, irrelevant to the comparison.

GPU:   python experiments/exp06_split_vs_full.py            (resumable: rerun to continue)
Smoke: python experiments/exp06_split_vs_full.py --smoke    (no torch; fake 4x model, 3 cropped pages)
"""
import argparse, json, os, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import cv2, numpy as np
from common import load_sample, read_page, line_mask, classify_tiles

TILE, SCALE = 128, 4
OUT = "outputs/exp06_split_vs_full"


# ---------------------------------------------------------------- routing
def edge_runs(kinds):
    """(row, col_start, n_tiles) for each horizontal run of edge tiles."""
    runs = []
    for r in range(kinds.shape[0]):
        c = 0
        while c < kinds.shape[1]:
            if kinds[r, c] == 2:
                s = c
                while c < kinds.shape[1] and kinds[r, c] == 2:
                    c += 1
                runs.append((r, s, c - s))
            else:
                c += 1
    return runs


def routed(src, kinds, margin, forward, vram_mb):
    h, w = src.shape[:2]
    H, W = -(-h // TILE) * TILE, -(-w // TILE) * TILE
    # pad to whole tiles + margin by reflecting real content (never a hard black edge)
    padded = cv2.copyMakeBorder(src, margin, H - h + margin, margin, W - w + margin,
                                cv2.BORDER_REFLECT_101)
    out = cv2.resize(src, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)
    runs = edge_runs(kinds)
    by_shape = {}
    for r, c, n in runs:
        by_shape.setdefault(n, []).append((r, c))
    px_sent = 0
    m4 = margin * SCALE
    for n, items in by_shape.items():
        ch, cw = TILE + 2 * margin, n * TILE + 2 * margin
        per_batch = max(1, int((0.75 * vram_mb - 32.1) / (0.0085 * ch * cw)))
        for b in range(0, len(items), per_batch):
            group = items[b:b + per_batch]
            crops = [padded[r * TILE:r * TILE + ch, c * TILE:c * TILE + cw] for r, c in group]
            ups = forward(crops)
            px_sent += len(crops) * ch * cw
            for (r, c), up in zip(group, ups):
                core = up[m4:m4 + TILE * SCALE, m4:m4 + n * TILE * SCALE]
                y0, x0 = r * TILE * SCALE, c * TILE * SCALE
                y1, x1 = min(y0 + core.shape[0], h * SCALE), min(x0 + core.shape[1], w * SCALE)
                out[y0:y1, x0:x1] = core[:y1 - y0, :x1 - x0]
    return out, px_sent, len(runs)


# ---------------------------------------------------------------- metrics
def ssim_map(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = g(a), g(b)
    va, vb, cov = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    return ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))


def region_scores(full2, test2, kinds):
    """SSIM over the whole page, edge tiles only, non-edge tiles only; plus PSNR."""
    smap = ssim_map(full2, test2)
    t2 = TILE * SCALE // 2
    em = np.kron((kinds == 2).astype(np.uint8), np.ones((t2, t2), np.uint8))
    em = em[:smap.shape[0], :smap.shape[1]].astype(bool)
    return {"ssim": float(smap.mean()),
            "ssim_edge": float(smap[em].mean()) if em.any() else 1.0,
            "ssim_rest": float(smap[~em].mean()) if (~em).any() else 1.0,
            "psnr": float(cv2.PSNR(full2, test2))}


def half(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)


def save_visual(path, full2, routed2, kinds):
    """600-row window with the most edge/non-edge borders: full | routed16 | difference x4."""
    t2 = TILE * SCALE // 2
    edge = (kinds == 2).astype(int)
    borders = np.abs(np.diff(edge, axis=1)).sum(1) if edge.shape[1] > 1 else np.zeros(edge.shape[0])
    r = int(np.argmax(np.convolve(borders, np.ones(4), "same")))
    y0 = max(0, r * t2 - 300)
    y1 = min(full2.shape[0], y0 + 600)
    diff = cv2.convertScaleAbs(cv2.absdiff(full2[y0:y1], routed2[y0:y1]), alpha=4)
    cv2.imwrite(path, np.hstack([full2[y0:y1], routed2[y0:y1], diff]))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pages = load_sample()

    if a.smoke:
        pages = pages[:3]
        vram = 8187
        def forward(crops):  # stand-in for the network: plain 4x enlargement
            return [cv2.resize(c, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
                    for c in crops]
        sync = lambda: None
        def run_full(src):
            return cv2.resize(src, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    else:
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
        vram = bmap["available_vram"]
        sync = torch.cuda.synchronize

        def forward(crops):
            x = torch.cat([_preprocess(c) for c in crops])
            xp, ph, pw = _pad(x, SCALE)
            with torch.no_grad():
                y = model(xp)
            y = _unpad(y, ph, pw, SCALE)
            return [_postprocess(t) for t in torch.split(y, 1, dim=0)]

        def run_full(src):
            chunks = chunk_page(src, 0, bmap)
            done = []
            for batch in build_batches(chunks, bmap):
                done.extend(batch_upscale(batch, model, SCALE))
            done.sort(key=lambda t: t[1].chunk_index)
            return reassemble(done, SCALE)

    if a.limit:
        pages = pages[:a.limit]
    results_path = os.path.join(OUT, "results_smoke.jsonl" if a.smoke else "results.jsonl")
    done_keys = set()
    if os.path.exists(results_path) and not a.smoke:
        with open(results_path) as f:
            done_keys = {json.loads(l)["key"] for l in f if l.strip()}
    log = open(results_path, "w" if a.smoke else "a")

    first = True
    for i, p in enumerate(pages):
        key = f'{p["cbz"]}::{p["page"]}'
        if key in done_keys:
            continue
        src = read_page(p["cbz"], p["page"])
        if a.smoke:
            src = src[:700]
        h, w = src.shape[:2]
        kinds = classify_tiles(src, line_mask(src), TILE)
        # first page of a process pays cuDNN autotune for new shapes; excluded from timing totals
        rec = {"key": key, "i": i, "series": p["series"], "size": f"{w}x{h}",
               "edge_tiles_pct": round(100 * float((kinds == 2).mean()), 1),
               "warm": not first}
        first = False
        sync(); t = time.perf_counter()
        full4 = run_full(src); sync()
        rec["t_full"] = round(time.perf_counter() - t, 2)
        full2 = half(full4)
        del full4
        r16 = None
        for m in (16, 32):
            sync(); t = time.perf_counter()
            r4, px, nruns = routed(src, kinds, m, forward, vram); sync()
            rec[f"t_r{m}"] = round(time.perf_counter() - t, 2)
            rec[f"px_r{m}_vs_page"] = round(px / (h * w), 3)
            rec["runs"] = nruns
            r2 = half(r4)
            del r4
            if m == 16:
                r16 = r2
            rec.update({f"{k}_r{m}": round(v, 4) for k, v in region_scores(full2, r2, kinds).items()})
        if i < 6 or a.smoke:
            save_visual(os.path.join(OUT, f"page{i:02d}_full_routed16_diff.png"), full2, r16, kinds)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f'{i + 1:2d}/{len(pages)} {p["series"][:20]:20s} full {rec["t_full"]:5.1f}s | '
              f'r16 {rec["t_r16"]:5.1f}s ssim {rec["ssim_r16"]:.4f} | '
              f'r32 {rec["t_r32"]:5.1f}s ssim {rec["ssim_r32"]:.4f}', flush=True)
    log.close()
    summarize(results_path)


def summarize(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    warm = [r for r in rows if r["warm"]] or rows
    tf = max(sum(r["t_full"] for r in warm), 1e-9)
    lines = [f"Pages: {len(rows)} ({len(warm)} warm, used for timing)",
             f"Full pipeline, warm pages: {tf:.0f}s"]
    for m in (16, 32):
        tr = sum(r[f"t_r{m}"] for r in warm)
        lines.append(
            f"Routed, margin {m}px: {tr:.0f}s ({100 * tr / tf:.0f}% of full) | "
            f"SSIM all {st.mean(r[f'ssim_r{m}'] for r in rows):.4f}, "
            f"edge {st.mean(r[f'ssim_edge_r{m}'] for r in rows):.4f}, "
            f"rest {st.mean(r[f'ssim_rest_r{m}'] for r in rows):.4f} | "
            f"PSNR {st.mean(r[f'psnr_r{m}'] for r in rows):.1f} dB | "
            f"worst page SSIM {min(r[f'ssim_r{m}'] for r in rows):.4f}")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(os.path.dirname(path), "summary.txt" if "smoke" not in path else "summary_smoke.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
