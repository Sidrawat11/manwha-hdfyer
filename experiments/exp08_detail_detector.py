"""Experiment 8: detail-based detector. CPU only.

Old rule (exp 2-6): a square goes to the network if it contains a thin, strong line.
New rule: ... OR if it contains enough fine detail of any kind, measured relative to its
brightness, so faint texture in dark scenes counts too.

Detail = original minus a blurred copy (only fine detail survives the subtraction).
Score  = spread of that leftover detail / (square brightness + 20).

Reports, for several thresholds:
  - share of squares sent to the network (60 pages)
  - network work vs today's pipeline (16 px margin, merged runs; same model as exp 3)
  - on the 4 pages that failed in exp 6: how many of their previously-blended squares get rescued
  - on the 3 pages whose blended areas scored best in exp 6: how many squares get pulled in
    anyway (squares that didn't need the network: pure cost)
"""
import os, sys, json, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2, numpy as np
from common import load_sample, read_page, line_mask, classify_tiles

TILE, MARGIN = 128, 16
THRESHOLDS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04)
FAIL = (14, 18, 32, 35)      # below 0.990 in exp 6
CLEAN = (39, 23, 26)         # best blended-area scores in exp 6
OUT = "outputs/exp08_detail_detector"

def detail_scores(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    h, w = gray.shape
    R, C = -(-h // TILE), -(-w // TILE)
    s = np.zeros((R, C), np.float32)
    for r in range(R):
        for c in range(C):
            ys, xs = slice(r*TILE, (r+1)*TILE), slice(c*TILE, (c+1)*TILE)
            s[r, c] = detail[ys, xs].std() / (gray[ys, xs].mean() + 20.0)
    return s

def today_cost(h, w, strip=898, ov=128):
    if h <= strip: return h * w
    n, y = 0, 0
    while True:
        n += 1
        if y + strip >= h: break
        y += strip - ov
    return (strip * (n - 1) + (h - y)) * w

def routed_cost(kinds, w, h):
    cost = 0
    for r in range(kinds.shape[0]):
        run, th = 0, min(TILE, h - r*TILE)
        for k in list(kinds[r]) + [0]:
            if k == 2: run += 1
            elif run:
                cost += (th + 2*MARGIN) * (min(run*TILE, w) + 2*MARGIN); run = 0
    return cost

def apply(old, scores, t):
    new = old.copy()
    new[(old == 1) & (scores > t)] = 2     # only smooth squares can be promoted; solid stays solid
    return new

def map_image(src, old, new, path):
    """Left: original. Right: blue = still blended, red = network (old rule), orange fill = newly rescued."""
    ov = src.copy(); h, w = src.shape[:2]
    for r in range(old.shape[0]):
        for c in range(old.shape[1]):
            ys, xs = slice(r*TILE, (r+1)*TILE), slice(c*TILE, (c+1)*TILE); t = ov[ys, xs]
            if new[r, c] == 1:
                ov[ys, xs] = cv2.addWeighted(t, 0.55, np.full_like(t, (255, 120, 0)), 0.45, 0)
            elif new[r, c] == 2 and old[r, c] == 1:
                ov[ys, xs] = cv2.addWeighted(t, 0.55, np.full_like(t, (0, 165, 255)), 0.45, 0)
            if new[r, c] == 2:
                cv2.rectangle(ov, (c*TILE, r*TILE), (min(w, (c+1)*TILE)-1, min(h, (r+1)*TILE)-1), (0, 0, 255), 2)
    out = np.hstack([src, np.full((h, 20, 3), 255, np.uint8), ov])
    cv2.imwrite(path, cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])

os.makedirs(OUT, exist_ok=True)
stats = {t: {"edge": [], "work": [], "rescued": [], "pulled": []} for t in THRESHOLDS}
old_edge, old_work, keep = [], [], {}
for i, p in enumerate(load_sample()):
    src = read_page(p["cbz"], p["page"]); h, w = src.shape[:2]
    old = classify_tiles(src, line_mask(src), TILE)
    sc = detail_scores(src)
    base = today_cost(h, w)
    old_edge.append((old == 2).mean()); old_work.append(routed_cost(old, w, h) / base)
    smooth = (old == 1)
    for t in THRESHOLDS:
        new = apply(old, sc, t)
        stats[t]["edge"].append((new == 2).mean())
        stats[t]["work"].append(routed_cost(new, w, h) / base)
        frac = float((smooth & (new == 2)).sum()) / max(1, smooth.sum())
        if i in FAIL: stats[t]["rescued"].append(frac)
        if i in CLEAN: stats[t]["pulled"].append(frac)
    if i in FAIL + CLEAN:
        keep[i] = (src, old, sc)

summary = {"old": {"edge_pct": 100*st.mean(old_edge), "work_pct": 100*st.mean(old_work)}}
print(f"Old line-only rule: {100*st.mean(old_edge):.0f}% of squares to network, work {100*st.mean(old_work):.0f}% of today")
print(f"{'threshold':>9} | {'squares':>7} | {'work':>5} | rescued on 4 failing pages | pulled in on 3 clean pages")
for t in THRESHOLDS:
    s = stats[t]
    summary[str(t)] = {k: 100*st.mean(v) for k, v in s.items()}
    print(f"{t:9.2f} | {100*st.mean(s['edge']):6.0f}% | {100*st.mean(s['work']):4.0f}% | "
          f"{100*st.mean(s['rescued']):5.0f}% of their blended squares | {100*st.mean(s['pulled']):5.0f}%")
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)

pick = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
for i, (src, old, sc) in keep.items():
    tag = "fail" if i in FAIL else "clean"
    map_image(src, old, apply(old, sc, pick), os.path.join(OUT, f"{tag}_page{i:02d}_t{pick}.jpg"))
print(f"maps written for threshold {pick}")
