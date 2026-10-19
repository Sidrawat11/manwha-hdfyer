"""Experiment 11: hybrid router, computed from experiment 10's measured timings. CPU only.

Rule: before upscaling, measure how much of the page the edge-routed path would send to the
network. Above a cut-off -> run today's full pipeline; below -> route. Busy pages pay nothing extra.
Two candidate measures:
  squares  share of 128 px squares classified edge (detail detector, 0.02)
  pixels   pixels the routed path would actually send INCLUDING 16 px margins,
           divided by the pixels today's strip pipeline sends (incl. its overlap)
Reports the best cut-off per measure, and a leave-one-series-out check: pick the cut-off on two
series, apply it to the third, so the cut-off isn't just tuned to the data it's judged on.
"""
import os, sys, json, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_sample, read_page, classify_detail

TILE, MARGIN = 128, 16
OUT = "outputs/exp11_hybrid_router"

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
        run, th = 0, min(TILE, h - r * TILE)
        for k in list(kinds[r]) + [0]:
            if k == 2: run += 1
            elif run:
                cost += (th + 2 * MARGIN) * (min(run * TILE, w) + 2 * MARGIN); run = 0
    return cost

rows = [json.loads(l) for l in open("outputs/exp10_detail_confirm/results.jsonl") if l.strip()]
sample = load_sample()
for r in rows:
    p = sample[r["i"]]; src = read_page(p["cbz"], p["page"]); h, w = src.shape[:2]
    k = classify_detail(src, TILE, 0.02)
    r["squares"] = r["squares_detail16"] / 100
    r["pixels"] = routed_cost(k, w, h) / today_cost(h, w)
timed = [r for r in rows if r["warm"] and not r["spill"]]

def hybrid(sel, measure, cut):
    tf = sum(r["t_full"] for r in sel)
    th = sum(r["t_full"] if r[measure] > cut else r["t_detail16"] for r in sel)
    routed = sum(r[measure] <= cut for r in sel)
    return th / tf, routed

cuts = [x / 100 for x in range(40, 131, 2)]
result = {"always_route": sum(r["t_detail16"] for r in timed) / sum(r["t_full"] for r in timed)}
print(f"Always route (exp 10): {100*result['always_route']:.0f}% of full time\n")
for measure in ("squares", "pixels"):
    best = min(cuts, key=lambda c: hybrid(timed, measure, c)[0])
    t, n = hybrid(timed, measure, best)
    result[measure] = {"cut": best, "time": t, "pages_routed": n}
    print(f"[{measure}] best cut-off {best:.2f}: {100*t:.0f}% of full time, {n}/{len(timed)} pages routed")
    for c in (best - 0.1, best - 0.05, best + 0.05, best + 0.1):
        t2, n2 = hybrid(timed, measure, c); print(f"     cut {c:.2f}: {100*t2:.0f}% ({n2} routed)")
    print("   leave-one-series-out:")
    for s in sorted({r['series'] for r in timed}):
        train = [r for r in timed if r["series"] != s]; test = [r for r in timed if r["series"] == s]
        c = min(cuts, key=lambda c: hybrid(train, measure, c)[0])
        t_test, _ = hybrid(test, measure, c)
        t_route = sum(r["t_detail16"] for r in test) / sum(r["t_full"] for r in test)
        print(f"     {s[:28]:28s} cut {c:.2f} -> {100*t_test:.0f}% (always-route {100*t_route:.0f}%)")
    print()
# where does routing stop paying? per-page ratio vs measure
print("Per page: routed/full time ratio by pixel measure bucket")
for lo, hi in ((0, .6), (.6, .8), (.8, .9), (.9, 1.0), (1.0, 2)):
    b = [r["t_detail16"] / r["t_full"] for r in timed if lo <= r["pixels"] < hi]
    if b: print(f"   pixels {lo:.1f}-{hi:.1f}: {len(b):2d} pages, routed takes {100*st.mean(b):.0f}% of full")
os.makedirs(OUT, exist_ok=True)
json.dump(result, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
