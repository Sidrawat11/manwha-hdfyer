"""Estimate network work (pixels incl. context margins) for tile routing vs today's strips."""
import os, sys, json, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
from common import load_sample, read_page, line_mask, classify_tiles

def today_cost(h, w, strip=898, ov=128):
    # rows actually sent through the network by the current chunker (approx: ignores single-chunk pages)
    if h <= strip: return h * w
    n, y = 0, 0
    while True:
        n += 1
        if y + strip >= h: break
        y += strip - ov
    return (strip * (n - 1) + (h - y)) * w

def routed_cost(kinds, tile, ctx, w, h):
    """Merge horizontally-adjacent edge tiles within each tile row into runs; pay context per run."""
    cost = 0
    for r in range(kinds.shape[0]):
        row = list(kinds[r]) + [0]
        run = 0
        th = min(tile, h - r * tile)
        for k in row:
            if k == 2: run += 1
            elif run:
                rw = min(run * tile, w)
                cost += (th + 2 * ctx) * (rw + 2 * ctx); run = 0
    return cost

res = {}
for p in load_sample():
    src = read_page(p["cbz"], p["page"]); h, w = src.shape[:2]; m = line_mask(src)
    base = today_cost(h, w)
    for tile in (32, 64, 128):
        k = classify_tiles(src, m, tile)
        for ctx in (16, 32):
            res.setdefault((tile, ctx), []).append(routed_cost(k, tile, ctx, w, h) / base)
print("Network work relative to today's pipeline (100% = today, incl. its strip overlap)")
for (tile, ctx), v in sorted(res.items()):
    print(f"  tile {tile:3d}, margin {ctx:2d}px: {100*st.mean(v):5.1f}%  (best page {100*min(v):5.1f}%, worst {100*max(v):5.1f}%)")

def band_cost(kinds, tile, band, ctx, w, h):
    """One crop per band of `band` rows: spans leftmost..rightmost edge tile; skipped if no edge tile."""
    per = band // tile; cost = 0
    for b0 in range(0, kinds.shape[0], per):
        blk = kinds[b0:b0 + per]
        cols = [c for c in range(blk.shape[1]) if (blk[:, c] == 2).any()]
        if not cols: continue
        bh = min(band, h - b0 * tile)
        bw = min((cols[-1] - cols[0] + 1) * tile, w)
        cost += (bh + 2 * ctx) * (bw + 2 * ctx)
    return cost

res2 = {}
for p in load_sample():
    src = read_page(p["cbz"], p["page"]); h, w = src.shape[:2]; m = line_mask(src)
    base = today_cost(h, w)
    k = classify_tiles(src, m, 32)
    for band in (64, 128, 256, 512):
        for ctx in (16, 32):
            res2.setdefault((band, ctx), []).append(band_cost(k, 32, band, ctx, w, h) / base)
print("\nBand routing (32px classification, one crop per band, full-width-ish)")
for (band, ctx), v in sorted(res2.items()):
    print(f"  band {band:3d} rows, margin {ctx:2d}px: {100*st.mean(v):5.1f}%  (best {100*min(v):5.1f}%, worst {100*max(v):5.1f}%)")
