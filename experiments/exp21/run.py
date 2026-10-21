"""Exp 21: sharpening bake-off on full pages, student (hdfyer raw) vs UltraSharp.

Every variant in sharpen.VARIANTS on both sources for the pages in pages.json; timing, scores, images and a
comparison page:

    experiments/exp21/scores.csv     one row per page x source x variant
    experiments/exp21/summary.csv    one row per source x variant (means over the pages, median ms per megapixel)
    outputs/exp21-sharpening/<nn-label>/<source>__<variant>.png, manifest.json, compare.html

Scores (on luma; geometry from plain UltraSharp, so both sources are measured at the same places):
    acutance        mean of the strongest gradient (levels/px) within 1 px of UltraSharp's Canny edges
    acutance_vs_us  acutance / plain UltraSharp's acutance on the same page (1.0 = as crisp as UltraSharp)
    halo_mean       mean overshoot (levels) in the 3 px band around edges: how far the variant goes outside the
                    min-max range of its own unsharpened source's 7x7 neighbourhood
    halo_frac       share of band pixels overshooting by more than 6 levels
    noise_gain      std of the high-pass (luma - gaussian 1.0) on flat areas / the same for the unsharpened source
    dist_us         mean absolute difference to plain UltraSharp (levels, all pixels and channels)
    dist_us_edges   the same on edges and the 3 px band

    python experiments/exp21/run.py              # everything (needs data/student/hdfyer-2x-s-r2-raw from make_student.py)
    python experiments/exp21/run.py --html-only  # rebuild compare.html from the saved manifest
"""
import argparse, csv, json, os, statistics, sys, time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, HERE)
os.chdir(ROOT)
from outputs import output_dir  # noqa: E402
import sharpen as S  # noqa: E402

SOURCES = {"student": "data/student/hdfyer-2x-s-r2-raw", "ultrasharp": "data/teacher/ultrasharp-2x"}
OUT = "outputs/exp21-sharpening"
FIELDS = ["page", "label", "source", "variant", "method", "mpx", "ms", "ms_per_mpx", "acutance", "acutance_vs_us",
          "halo_mean", "halo_frac", "noise_gain", "dist_us", "dist_us_edges"]


def y_of(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def grad(y):
    return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)) / 8


def edge_grad(y):
    """Strongest gradient within 1 px of each pixel: acutance that a sub-pixel edge shift (warp) does not penalise."""
    return cv2.dilate(grad(y), np.ones((3, 3), np.uint8))


def highpass_std(y, mask):
    return float((y - cv2.GaussianBlur(y, (0, 0), 1.0))[mask].std()) if mask.any() else float("nan")


class Geometry:
    """Where to measure on one page, from plain UltraSharp."""

    def __init__(self, us):
        y = y_of(us)
        self.edges = cv2.Canny(us, 60, 150) > 0
        near = cv2.dilate(self.edges.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        self.band = near & ~self.edges
        self.near = near
        mu = cv2.blur(y, (9, 9)); var = cv2.blur(y * y, (9, 9)) - mu * mu
        self.flat = (var < 4) & ~cv2.dilate(near.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        self.us = us.astype(np.int16)
        self.us_acut = float(edge_grad(y)[self.edges].mean())


class SourceRef:
    """The unsharpened source's local range (for overshoot) and flat-area noise (for noise gain)."""

    def __init__(self, src, geo):
        y = y_of(src)
        k = np.ones((7, 7), np.uint8)
        self.lo, self.hi = cv2.erode(y, k), cv2.dilate(y, k)
        self.noise = highpass_std(y, geo.flat)


def score(out, geo, ref):
    y = y_of(out)
    over = np.maximum(y - ref.hi, 0) + np.maximum(ref.lo - y, 0)
    ob = over[geo.band]
    acut = float(edge_grad(y)[geo.edges].mean())
    d = np.abs(out.astype(np.int16) - geo.us)
    return {"acutance": round(acut, 3), "acutance_vs_us": round(acut / geo.us_acut, 4),
            "halo_mean": round(float(ob.mean()), 4), "halo_frac": round(float((ob > 6).mean()), 5),
            "noise_gain": round(highpass_std(y, geo.flat) / ref.noise, 4) if ref.noise > 0 else float("nan"),
            "dist_us": round(float(d.mean()), 4), "dist_us_edges": round(float(d[geo.near].mean()), 4)}


def timed(fn, img, kw):
    if S.DEV == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(img, **kw)
    if S.DEV == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()
    if a.html_only:
        data = json.load(open(os.path.join(OUT, "manifest.json")))["compare"]
        write_html(data)
        return
    pages = json.load(open(os.path.join(HERE, "pages.json")))["pages"]
    for p in pages:
        for src, d in SOURCES.items():
            if not os.path.exists(os.path.join(d, p["page"])):
                sys.exit(f"missing {src} page {p['page']}: run make_student.py first")
    output_dir(OUT, note="Exp 21 sharpening bake-off; page images, local only")
    warm = cv2.imread(os.path.join(SOURCES["ultrasharp"], pages[0]["page"]), cv2.IMREAD_COLOR)[:512]
    for fn, kw in S.VARIANTS.values():  # first call of each (CUDA kernels, cudnn) outside the timing
        fn(warm, **kw)
    rows, compare = [], []
    for n, p in enumerate(pages, 1):
        pid = f"{n:02d}-{p['label']}"
        os.makedirs(os.path.join(OUT, pid), exist_ok=True)
        srcs = {s: cv2.imread(os.path.join(d, p["page"]), cv2.IMREAD_COLOR) for s, d in SOURCES.items()}
        geo = Geometry(srcs["ultrasharp"])
        mpx = srcs["ultrasharp"].shape[0] * srcs["ultrasharp"].shape[1] / 1e6
        entry = {"id": pid, "label": p["label"], "content": p["content"], "w": srcs["ultrasharp"].shape[1],
                 "h": srcs["ultrasharp"].shape[0], "variants": []}
        for s, img in srcs.items():
            ref = SourceRef(img, geo)
            for name, (fn, kw) in S.VARIANTS.items():
                out, ms = timed(fn, img, kw)
                f = f"{s}__{name}.png"
                cv2.imwrite(os.path.join(OUT, pid, f), out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                rows.append({"page": pid, "label": p["label"], "source": s, "variant": name, "method": fn.__name__,
                             "mpx": round(mpx, 2), "ms": round(ms, 1), "ms_per_mpx": round(ms / mpx, 2), **score(out, geo, ref)})
                entry["variants"].append({"key": f"{s} · {name}", "file": f"{pid}/{f}"})
            print(f"[{n:2d}/{len(pages)}] {pid} {s}: done", flush=True)
        compare.append(entry)
        del srcs, geo
    with open(os.path.join(HERE, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    summary = summarize(rows)
    with open(os.path.join(HERE, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0])); w.writeheader(); w.writerows(summary)
    output_dir(OUT, note="Exp 21 sharpening bake-off; page images, local only", sources=SOURCES,
               variants={k: {"method": fn.__name__, **kw} for k, (fn, kw) in S.VARIANTS.items()},
               pages="experiments/exp21/pages.json", device=torch.cuda.get_device_name(0) if S.DEV == "cuda" else "cpu",
               compare=compare)
    write_html(compare)
    print_summary(summary)


def summarize(rows):
    out = []
    keys = []
    for r in rows:
        if (r["source"], r["variant"]) not in keys:
            keys.append((r["source"], r["variant"]))
    for s, v in keys:
        rs = [r for r in rows if r["source"] == s and r["variant"] == v]
        m = lambda k: round(statistics.fmean(r[k] for r in rs), 4)
        out.append({"source": s, "variant": v, "method": rs[0]["method"], "pages": len(rs),
                    "ms_per_mpx_median": round(statistics.median(r["ms_per_mpx"] for r in rs), 2),
                    **{k: m(k) for k in ("acutance", "acutance_vs_us", "halo_mean", "halo_frac", "noise_gain", "dist_us", "dist_us_edges")}})
    return out


def print_summary(summary):
    print(f"\n{'source':10s} {'variant':20s} {'ms/Mpx':>7s} {'acut/US':>8s} {'halo':>6s} {'halo>6':>7s} {'noise':>6s} {'dist US':>8s} {'edges':>6s}")
    for r in summary:
        print(f"{r['source']:10s} {r['variant']:20s} {r['ms_per_mpx_median']:7.1f} {r['acutance_vs_us']:8.3f} {r['halo_mean']:6.2f} "
              f"{r['halo_frac'] * 100:6.2f}% {r['noise_gain']:6.2f} {r['dist_us']:8.3f} {r['dist_us_edges']:6.2f}")


def write_html(compare):
    html = HTML.replace("__DATA__", json.dumps(compare))
    open(os.path.join(OUT, "compare.html"), "w").write(html)
    print(f"wrote {os.path.join(OUT, 'compare.html')}")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Sharpening Bake-off</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#121212;--panel:#1d1d1d;--line:#333;--text:#e4e4e4;--dim:#9a9a9a;--accent:#5b9bff;--left:#ffb347;--right:#6fd3ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:8px 12px;
 display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
select,button{background:#262626;color:var(--text);border:1px solid #444;border-radius:6px;padding:5px 8px;font:inherit}
button{cursor:pointer}button:hover{border-color:var(--accent)}button.on{background:#24406b;border-color:var(--accent)}
label{color:var(--dim)}.l{color:var(--left)}.r{color:var(--right)}input[type=range]{width:200px}
#meta{color:var(--dim);width:100%}#picked{color:var(--dim)}
main{padding:12px;overflow-x:auto}
#stack{position:relative;margin:0 auto;cursor:ew-resize;user-select:none}
#stack img{display:block;position:absolute;top:0;left:0;width:100%;height:auto}
#stack img.px{image-rendering:pixelated}
#imgR{position:relative!important}
#bar{position:absolute;top:0;bottom:0;width:2px;background:#fff;box-shadow:0 0 0 1px #000;pointer-events:none}
.tag{position:sticky;top:96px;z-index:3;display:inline-block;padding:3px 8px;border-radius:4px;background:rgba(0,0,0,.75);font-weight:600}
#tags{display:flex;justify-content:space-between;position:sticky;top:96px;z-index:4;pointer-events:none;height:0}
</style></head><body>
<header>
 <label>Page <select id="page"></select></label>
 <label class="l">Left <select id="left"></select></label>
 <button id="swap" title="Swap left and right">⇄</button>
 <label class="r">Right <select id="right"></select></label>
 <label>Split <input id="split" type="range" min="0" max="100" value="50"></label>
 <label><input id="follow" type="checkbox" checked> follow mouse</label>
 <button id="zoom">2× zoom</button>
 <span>Winner:</span><button id="pl" class="l">Left</button><button id="ps">Same</button><button id="pr" class="r">Right</button>
 <button id="dl">Download picks</button><span id="picked"></span>
 <div id="meta"></div>
</header>
<main><div id="tags"><span class="tag l" id="tagL"></span><span class="tag r" id="tagR"></span></div>
<div id="stack"><img id="imgR" alt="right variant"><img id="imgL" alt="left variant"><div id="bar"></div></div></main>
<script>
const D=__DATA__;const $=id=>document.getElementById(id);
let picks={};try{picks=JSON.parse(localStorage.getItem('exp21-picks')||'{}')}catch(e){}
let zoom=false;
D.forEach((p,i)=>$('page').add(new Option(`${p.id}  (${p.content.join(', ')})`,i)));
function cur(){return D[+$('page').value]}
function fill(){const p=cur(),l=$('left').value,r=$('right').value;
 for(const s of [$('left'),$('right')]){s.innerHTML='';p.variants.forEach(v=>s.add(new Option(v.key,v.key)));}
 $('left').value=p.variants.some(v=>v.key===l)?l:'student · baseline';
 $('right').value=p.variants.some(v=>v.key===r)?r:'ultrasharp · plain';}
function file(k){return cur().variants.find(v=>v.key===k).file}
function show(){const p=cur();$('imgL').src=file($('left').value);$('imgR').src=file($('right').value);
 const w=p.w/(window.devicePixelRatio||1)*(zoom?2:1);$('stack').style.width=w+'px';
 for(const im of [$('imgL'),$('imgR')])im.classList.toggle('px',zoom);
 $('tagL').textContent='◀ '+$('left').value;$('tagR').textContent=$('right').value+' ▶';
 $('meta').textContent=`${p.id}: ${p.w}×${p.h} px, shown 1:1 device pixels${zoom?' ×2':''}. Content: ${p.content.join(', ')}`;
 split();showPick();}
function split(){const v=+$('split').value;$('imgL').style.clipPath=`inset(0 ${100-v}% 0 0)`;$('bar').style.left=v+'%';}
function pickKey(){return `${cur().id}|${$('left').value}|${$('right').value}`}
function pick(w){const winner=w==='same'?'same':(w==='left'?$('left').value:$('right').value);
 picks[pickKey()]={page:cur().id,left:$('left').value,right:$('right').value,winner,time:new Date().toISOString()};
 try{localStorage.setItem('exp21-picks',JSON.stringify(picks))}catch(e){}showPick();}
function showPick(){const p=picks[pickKey()];$('picked').textContent=(p?`picked: ${p.winner}`:'')+` · ${Object.keys(picks).length} picks saved`;}
$('stack').addEventListener('mousemove',e=>{if(!$('follow').checked)return;const b=$('stack').getBoundingClientRect();
 $('split').value=Math.max(0,Math.min(100,(e.clientX-b.left)/b.width*100));split();});
$('split').oninput=split;$('page').onchange=()=>{fill();show();};$('left').onchange=show;$('right').onchange=show;
$('swap').onclick=()=>{const l=$('left').value;$('left').value=$('right').value;$('right').value=l;show();};
$('zoom').onclick=()=>{zoom=!zoom;$('zoom').classList.toggle('on',zoom);show();};
$('pl').onclick=()=>pick('left');$('pr').onclick=()=>pick('right');$('ps').onclick=()=>pick('same');
$('dl').onclick=()=>{const b=new Blob([JSON.stringify(Object.values(picks),null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='exp21-picks.json';a.click();};
window.addEventListener('resize',show);fill();show();
</script></body></html>"""

if __name__ == "__main__":
    main()
