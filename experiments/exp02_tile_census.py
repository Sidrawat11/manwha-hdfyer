"""Tile census over the shared 60-page sample. CPU only."""
import os, sys, json, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import build_sample, read_page, line_mask, classify_tiles

sample = build_sample()
rows = []
for i, p in enumerate(sample):
    src = read_page(p["cbz"], p["page"])
    h, w = src.shape[:2]
    mask = line_mask(src)
    rec = {"i": i, "series": p["series"], "chapter": os.path.basename(p["cbz"]),
           "page": p["page"], "size": f"{w}x{h}", "line_pct": 100 * float((mask > 0).mean())}
    for tile in (128, 64, 32):
        k = classify_tiles(src, mask, tile)
        n = k.size
        rec[f"edge{tile}"] = 100 * float((k == 2).sum()) / n
        rec[f"solid{tile}"] = 100 * float((k == 0).sum()) / n
        rec[f"smooth{tile}"] = 100 * float((k == 1).sum()) / n
    rows.append(rec)
    print(f"{i+1:2d}/60 {p['series'][:22]:22s} edge64={rec['edge64']:5.1f}%", flush=True)

with open("experiments/out/exp02_tile_census.json", "w") as f:
    json.dump(rows, f, indent=1)

def summary(sel, key):
    v = [r[key] for r in sel]
    return f"{st.mean(v):5.1f} ±{(st.stdev(v)/len(v)**0.5 if len(v)>1 else 0):4.1f}  (min {min(v):4.1f}, max {max(v):4.1f})"

print("\nEdge share = fraction of tiles that would still need the network (mean ± standard error)")
for series in sorted({r["series"] for r in rows}) + ["ALL"]:
    sel = rows if series == "ALL" else [r for r in rows if r["series"] == series]
    print(f"\n{series} (n={len(sel)})")
    for t in (128, 64, 32):
        print(f"  tile {t:3d}: edge {summary(sel, f'edge{t}')} | solid {summary(sel, f'solid{t}')}")
    print(f"  line pixels: {summary(sel, 'line_pct')}")
