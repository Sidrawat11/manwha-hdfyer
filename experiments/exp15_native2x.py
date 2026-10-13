"""Experiment 15: native 2x models vs today's 4x-then-halve.

Today the network draws every page at 4x, then we Lanczos-halve it to 2x: three quarters of the
work is thrown away. A model trained for 2x skips that. But it is a DIFFERENT model, so it draws
differently: this experiment is mostly a quality judgement by eye, with speed measured alongside.

Reference ("Today"): 4x-UltraSharp, real strip planner + reassembler, eager FP16, Lanczos half.
Candidates (loaded through spandrel, any that exist in models/ are used):
  FDAT detail   2x IllustrationJaNai V3detail FDAT M unshuffle   (CC BY-NC 4.0)
  FDAT denoise  2x IllustrationJaNai V3denoise FDAT M unshuffle  (CC BY-NC 4.0)
  SPAN detail   2x IllustrationJaNai V3detail SPAN S             (CC BY-NC 4.0)
  SPAN denoise  2x IllustrationJaNai V3denoise SPAN S            (CC BY-NC 4.0)
  DeJPEG        2x IllustrationJaNai V2 DeJPEG FDAT M            (CC BY-NC 4.0)
  HFA2k Compact 2x HFA2kCompact                                  (CC-BY-4.0, commercial OK)
All raw output, no sharpen/cleanup (same as every earlier experiment).

Candidates run in fixed 512-row strips, 32-row overlap, batch 1, stitched with the real reassembler.
Our VRAM formula is RRDBNet-only, so on an out-of-memory a page retries at 256, then 128 rows.
If FP16 produces NaN/inf in a strip (a known risk for attention models) that strip reruns in FP32
and the page is flagged.

Pages: the same 10 as Exp 14 (experiments/out/exp14_tensorrt/pages.json); page 6 is the rough
low-quality source where 4x-UltraSharp smears.

Outputs (experiments/out/exp15_native2x/):
  results.jsonl, summary.txt
  compare_pageNN.html   buttons: Original / Today / each model, plus a slider Today vs the last model picked.
                        Opens at actual pixel size ("Fit width" to toggle).

Setup once (in the venv):  pip install spandrel
Smoke test (1 page):       python experiments/exp15_native2x.py --smoke
Full run:                  python experiments/exp15_native2x.py
Only some models:          --models span_detail,hfa2k
Whole pages in the comparer (big files, ~30-40 MB each): --full
Resumable; a STOP file in the out folder stops it cleanly between pages.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, base64, gc, json, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import cv2, numpy as np
from tqdm import tqdm
from common import read_page, detail_scores
import exp06_split_vs_full as e6

OUT = "experiments/out/exp15_native2x"
PAGES = "experiments/out/exp14_tensorrt/pages.json"
STRIP, OVERLAP = 512, 32
MODE_NOTE = "All versions are raw output (no sharpening)."

CANDIDATES = [  # key, button label, file in models/
    ("fdat_detail", "FDAT detail", "2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.safetensors"),
    ("fdat_denoise", "FDAT denoise", "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.safetensors"),
    ("span_detail", "SPAN detail", "2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.safetensors"),
    ("span_denoise", "SPAN denoise", "2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16.safetensors"),
    ("dejpeg", "DeJPEG", "2x_IllustrationJaNai-DeJPEG_FDAT_M_100k.safetensors"),
    ("hfa2k", "HFA2k Compact", "2xHFA2kCompact.safetensors"),
]


# ---------------------------------------------------------------- comparer page
HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 15 page {n}</title><style>
:root{{color-scheme:dark}} body{{margin:0;background:#111;color:#ddd;font:15px system-ui,sans-serif}}
header{{padding:10px 14px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;position:sticky;top:0;background:#111;z-index:3}}
header b{{flex:1 1 100%}} button{{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid #555;background:#222;color:#eee}}
button.on{{border-color:#c9a227;color:#f0d878}} button.cmp{{border-style:dashed}}
.scroll{{overflow-x:auto}} .wrap{{position:relative;width:max-content;user-select:none}} .fit .wrap{{width:100%}} .fit .wrap img{{width:100%}}
.wrap img{{display:block;height:auto;max-width:none}} .over{{position:absolute;inset:0;width:100%;height:100%}}
#bar{{position:absolute;top:0;bottom:0;width:2px;background:#c9a227;pointer-events:none}}
p{{padding:0 14px;color:#aaa;line-height:1.5}} table{{margin:0 14px 20px;border-collapse:collapse}} td,th{{padding:4px 10px;border-bottom:1px solid #333;text-align:left}}
</style></head><body>
<header><b>Page {n} &middot; {info}</b>
{buttons}<button id="b_s">Slider</button><button id="b_z">Fit width</button>
<input id="rng" type="range" min="0" max="100" value="50" step="0.1" style="flex:1 1 220px"></header>
<div class="scroll"><div class="wrap" id="w"><img id="imgBase"><img id="imgOver" class="over"><div id="bar"></div></div></div>
<p id="cap"></p>
<p>Buttons show one version full width. Slider: left of the gold line is Today, right is the model you picked last
(dashed border). Original = the source page enlarged 2x with no smoothing, so you can see what was really there.
{mode_note}</p>
<table><tr><th>Version</th><th>Seconds for this page</th><th>vs Today</th><th>Similarity to Today</th></tr>{rows}</table>
<script>
const SRC={src};const LABEL={labels};
const imgBase=document.getElementById("imgBase"),imgOver=document.getElementById("imgOver"),bar=document.getElementById("bar"),w=document.getElementById("w"),cap=document.getElementById("cap");
let mode="s",pos=50,down=false,cur="{first}";
function clip(){{imgOver.style.clipPath="inset(0 "+(100-pos)+"% 0 0)";bar.style.left=pos+"%";const g=document.getElementById("rng");if(g)g.value=pos;}}
function marks(){{for(const k in SRC){{const b=document.getElementById("b_"+k);b.classList.toggle("on",k===mode);b.classList.toggle("cmp",k===cur&&mode==="s");}}
 document.getElementById("b_s").classList.toggle("on",mode==="s");}}
function show(m){{mode=m;
 if(m==="s"){{imgBase.src=SRC[cur];imgOver.src=SRC.today;imgOver.style.display="block";bar.style.display="block";clip();cap.textContent="Slider: Today | "+LABEL[cur];}}
 else{{imgBase.src=SRC[m];imgOver.style.display="none";bar.style.display="none";cap.textContent="Showing: "+LABEL[m];if(m!=="today"&&m!=="orig")cur=m;}}
 marks();}}
function at(ev){{if(mode!=="s")return;const R=w.getBoundingClientRect();pos=Math.max(0,Math.min(100,(ev.clientX-R.left)/R.width*100));clip();}}
w.addEventListener("pointerdown",ev=>{{if(ev.pointerType!=="mouse")return;down=true;at(ev);}});addEventListener("pointerup",()=>{{down=false;}});
w.addEventListener("pointermove",ev=>{{if(down)at(ev);}});
for(const k in SRC)document.getElementById("b_"+k).onclick=()=>show(k);
document.getElementById("b_s").onclick=()=>show("s");
const rng=document.getElementById("rng"),bz=document.getElementById("b_z");let fit=false;
rng.oninput=()=>{{pos=+rng.value;clip();}};
function size(){{const r=window.devicePixelRatio||1;imgBase.style.width=fit?"":(imgBase.naturalWidth/r)+"px";bz.textContent=fit?"Actual pixels":"Fit width";document.body.classList.toggle("fit",fit);}}
imgBase.addEventListener("load",size);bz.onclick=()=>{{fit=!fit;size();}};
show("s");
</script></body></html>"""


def jpg64(img, q=95):
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def write_compare(path, n, info, src, versions, labels, rec, full=False):
    """versions: ordered {key: 2x image}, must include 'today'. Crop = most detailed ~900 output rows, full width;
    full=True embeds the whole page instead (big file)."""
    h = src.shape[0]
    if full:
        y0, y1 = 0, min(h, min(v.shape[0] for v in versions.values()) // 2)
    else:
        sc = detail_scores(src, 128).mean(axis=1)
        k = 450 // 128 + 1
        best = int(np.argmax(np.convolve(sc, np.ones(k), "valid"))) if len(sc) > k else 0
        y0 = min(best * 128, max(0, h - 450)); y1 = min(h, y0 + 450)
    imgs = {"orig": cv2.resize(src[y0:y1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)}
    for key, im in versions.items():
        imgs[key] = im[2 * y0:2 * y1]
    labels = {"orig": "Original (2x, no smoothing)", **labels}
    buttons = "".join(f'<button id="b_{k}">{labels[k] if k != "orig" else "Original"}</button>' for k in imgs)
    rows = ""
    for key in versions:
        t = rec.get(f"t_{key}")
        sp = f'{rec["t_today"] / t:.1f}x faster' if key != "today" and t else ""
        ss = f'{rec[f"ssim_{key}"]:.4f}' if f"ssim_{key}" in rec else ""
        flag = " (some strips in FP32)" if rec.get(f"fp32_{key}") else ""
        rows += f"<tr><td>{labels[key]}{flag}</td><td>{t if t else ''}</td><td>{sp}</td><td>{ss}</td></tr>"
    first = next((k for k in versions if k != "today"), "today")
    src_js = json.dumps({k: jpg64(v, 90 if full else 95) for k, v in imgs.items()})
    open(path, "w").write(HTML.format(mode_note=MODE_NOTE, n=n, info=info, buttons=buttons, rows=rows, src=src_js,
                                      labels=json.dumps(labels), first=first))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shipped", action="store_true", help="apply the pipeline's sharpen + cleanup to every version (what the reader sees)")
    ap.add_argument("--full", action="store_true", help="also write compare_pageNN_full.html with the whole page")
    ap.add_argument("--models", default="", help="comma-separated keys, default: every candidate file found")
    a = ap.parse_args()

    import torch
    import torch.backends.cudnn as cudnn
    from spandrel import ModelLoader
    from core.model_loader import load_rrdbnet
    from profiler.cache import load_cache
    from inference.chunker import ChunkMeta, get_strip_height, plan_strips, get_max_chunk_height
    from inference.engine import _preprocess, _pad, _unpad, _postprocess
    from pipeline.reassembler import reassemble
    cudnn.benchmark = True
    sync = torch.cuda.synchronize
    bmap = load_cache()

    # ---- today's reference
    rrdb = load_rrdbnet("models/4x-UltraSharp.pth")

    def stitch(ups, plan, scale):
        n = len(plan)
        metas = [ChunkMeta(0, None, k, n, a0, b0, plan[k - 1][1] - a0 if k else 0, b0 - plan[k + 1][0] if k < n - 1 else 0)
                 for k, (a0, b0) in enumerate(plan)]
        return reassemble(list(zip(ups, metas)), scale)

    def today(src):
        h, w = src.shape[:2]
        plan = plan_strips(h, get_strip_height(w, bmap), max_strip_height=get_max_chunk_height(w, bmap))
        ups = []
        for a0, b0 in plan:
            x = _preprocess(src[a0:b0]); xp, ph, pw = _pad(x, 4)
            with torch.no_grad():
                y = rrdb(xp)
            ups.append(_postprocess(_unpad(y, ph, pw, 4)))
            del x, xp, y
        return stitch(ups, plan, 4)  # 4x; halved (and sharpened with --shipped) outside the timer

    # ---- candidates
    want = {k for k in a.models.split(",") if k}
    from postprocessing.sharpen import sharpen
    from postprocessing.cleanup import cleanup_near_black

    def finish(img, scale):
        """--shipped: the pipeline's finishing steps. Today sharpens at 4x (radius 1.0) before halving;
        a 2x model gets radius 0.5, the same blur width measured on the final 2x page."""
        if not a.shipped:
            return img
        return cleanup_near_black(sharpen(img, radius=1.0 if scale == 4 else 0.5))

    models, labels = {}, {"today": "Today (4x-UltraSharp, halved)"}
    for key, label, fname in CANDIDATES:
        if want and key not in want:
            continue
        path = os.path.join("models", fname)
        if not os.path.exists(path):
            print(f"  skipping {label}: {path} not found", flush=True)
            continue
        desc = ModelLoader().load_from_file(path)
        if desc.scale != 2:
            print(f"  skipping {label}: scale is {desc.scale}, expected 2", flush=True)
            continue
        desc.cuda().half().eval()
        models[key] = {"desc": desc, "fp32": None, "path": path}
        labels[key] = label
        print(f"  loaded {label}: {desc.architecture.name}, {sum(p.numel() for p in desc.model.parameters()) / 1e6:.1f}M params", flush=True)
    if not models:
        sys.exit("No candidate models found in models/. See the docstring for file names.")

    def run_strip(key, xs):
        """xs: RGB fp16 NCHW in [0,1]. Returns 2x fp16 tensor; reruns in FP32 if FP16 gives NaN/inf."""
        m = models[key]
        y = m["desc"](xs)
        if torch.isfinite(y).all():
            return y, False
        if m["fp32"] is None:
            m["fp32"] = ModelLoader().load_from_file(m["path"]).cuda().float().eval()
        return m["fp32"](xs.float()).half(), True

    def candidate(key, src):
        h, w = src.shape[:2]
        for strip in (STRIP, 256, 128):
            try:
                plan = plan_strips(h, min(strip, h), OVERLAP)
                ups, fp32 = [], False
                for a0, b0 in plan:
                    x = _preprocess(src[a0:b0])
                    y, used32 = run_strip(key, x)
                    fp32 |= used32
                    ups.append(_postprocess(y))
                    del x, y
                return stitch(ups, plan, 2), strip, fp32
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() or strip == 128:
                    raise
                print(f"    {labels[key]}: out of memory at {strip}-row strips, retrying smaller", flush=True)
                gc.collect(); torch.cuda.empty_cache()

    global OUT, MODE_NOTE
    if a.shipped:
        OUT = OUT + "_shipped"
        MODE_NOTE = "All versions have the pipeline's sharpening and cleanup applied (what the reader sees)."
    pages = json.load(open(PAGES))
    if a.smoke:
        pages = pages[:1]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results_smoke.jsonl" if a.smoke else "results.jsonl")
    stop = os.path.join(OUT, "STOP")
    if os.path.exists(stop): os.remove(stop)
    rows = []
    if os.path.exists(path) and not a.smoke:
        for l in open(path):
            try: rows.append(json.loads(l))
            except ValueError: pass
    done = {r["i"] for r in rows}
    log = open(path, "w" if a.smoke else "a")
    warmed = set()
    bar = tqdm(total=len(pages), initial=len(done & set(range(len(pages)))), unit="page", desc="Exp 15")
    for i, p in enumerate(pages):
        if i in done: continue
        if os.path.exists(stop):
            tqdm.write("STOP file found: stopping cleanly. Rerun to resume."); break
        src = read_page(p["cbz"], p["page"]); h, w = src.shape[:2]
        runs = [("today", today)] + [(k, (lambda s, k=k: candidate(k, s)[0])) for k in models]
        for name, fn in runs:  # warm-up per (model, width): cuDNN tuning is a one-off cost, not timed
            if (name, w) not in warmed:
                t = time.perf_counter(); fn(src); sync()
                tqdm.write(f"  warm-up {labels[name]} at width {w}: {time.perf_counter() - t:.0f}s")
                warmed.add((name, w))
        rec = {"i": i, "series": p["series"], "chapter": os.path.basename(p["cbz"]), "page": p["page"], "size": f"{w}x{h}"}
        outs = {}
        gc.collect(); torch.cuda.empty_cache()
        sync(); t = time.perf_counter(); o4 = today(src); sync()
        rec["t_today"] = round(time.perf_counter() - t, 2)
        outs["today"] = e6.half(finish(o4, 4)); del o4
        for key in models:
            gc.collect(); torch.cuda.empty_cache()
            sync(); t = time.perf_counter(); o, strip, fp32 = candidate(key, src); sync()
            rec[f"t_{key}"] = round(time.perf_counter() - t, 2)
            o = finish(o, 2)
            rec[f"strip_{key}"] = strip
            if fp32: rec[f"fp32_{key}"] = True
            if o.shape != outs["today"].shape:  # Lanczos half floors odd sizes; align before comparing
                hh, ww = min(o.shape[0], outs["today"].shape[0]), min(o.shape[1], outs["today"].shape[1])
                o = o[:hh, :ww]; outs["today"] = outs["today"][:hh, :ww]
            outs[key] = o
            rec[f"ssim_{key}"] = round(float(e6.ssim_map(outs["today"], o).mean()), 4)
            rec[f"meandiff_{key}"] = round(float(cv2.absdiff(outs["today"], o).mean()), 2)
        info = f'{p["series"]} &middot; {os.path.basename(p["cbz"])} &middot; {p["page"]}'
        write_compare(os.path.join(OUT, f"compare_page{i:02d}.html"), i, info, src, outs, labels, rec)
        if a.full:
            write_compare(os.path.join(OUT, f"compare_page{i:02d}_full.html"), i, info, src, outs, labels, rec, full=True)
        log.write(json.dumps(rec) + "\n"); log.flush(); rows.append(rec)
        line = f'{len(rows):2d}/{len(pages)} page {i:2d} {p["series"][:18]:18s} today {rec["t_today"]:5.1f}s'
        for key in models:
            line += f' | {key} {rec["t_" + key]:4.1f}s'
        tqdm.write(line)
        del outs, src; gc.collect(); torch.cuda.empty_cache()
        bar.update(1)
    bar.close()
    log.close()
    if not rows: return
    base = sum(r["t_today"] for r in rows)
    lines = [f"{len(rows)} pages (warm-ups excluded). Today (4x-UltraSharp, halved): {base / len(rows):.1f}s per page",
             "Similarity to Today is NOT a quality score here: a different model is meant to look different. Judge by eye."]
    for key in models:
        rs = [r for r in rows if f"t_{key}" in r]
        if not rs: continue
        tk = sum(r[f"t_{key}"] for r in rs); tb = sum(r["t_today"] for r in rs)
        fp = sum(1 for r in rs if r.get(f"fp32_{key}"))
        strips = sorted({r[f"strip_{key}"] for r in rs})
        lines.append(f"{labels[key]:14s}: {tk / len(rs):5.1f}s per page, {tb / tk:4.1f}x faster than Today | "
                     f"similarity to Today {st.mean(r[f'ssim_{key}'] for r in rs):.4f}, mean diff "
                     f"{st.mean(r[f'meandiff_{key}'] for r in rs):.1f} levels | strip rows {strips}"
                     + (f" | FP32 fallback on {fp} page(s)" if fp else ""))
    text = "\n".join(lines); print("\n" + text)
    open(os.path.join(OUT, "summary_smoke.txt" if a.smoke else "summary.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
