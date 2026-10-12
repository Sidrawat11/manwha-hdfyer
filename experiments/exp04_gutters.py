"""Share of rows in full-width solid runs (panel gutters) over the 60-page sample. CPU only."""
import os, sys, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import load_sample, read_page

def solid_rows(img, tol=8):
    spread = img.max(axis=1).astype(np.int16) - img.min(axis=1).astype(np.int16)
    return spread.max(axis=1) <= tol

def rows_in_runs(mask, min_len):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    lens = np.where(d == -1)[0] - np.where(d == 1)[0]
    return int(lens[lens >= min_len].sum())

res = {}
for p in load_sample():
    img = read_page(p["cbz"], p["page"]); m = solid_rows(img)
    for k in (32, 64, 128):
        res.setdefault((p["series"], k), []).append(100 * rows_in_runs(m, k) / len(m))
for s in sorted({s for s, _ in res}) + ["ALL"]:
    print(s)
    for k in (32, 64, 128):
        v = [x for (ss, kk), xs in res.items() if kk == k and (s == "ALL" or ss == s) for x in xs]
        print(f"  runs >= {k:3d} rows: {st.mean(v):5.1f}% of rows (min {min(v):4.1f}, max {max(v):4.1f})")
