"""Experiment 14: same model, faster runtime. Eager PyTorch vs torch.compile vs TensorRT FP16.

The maths is the same in all three, so any quality change comes only from rounding/kernel choice.
Strips come from the real pipeline planner (inference.chunker.get_strip_height / plan_strips):
fixed shapes are what make TensorRT practical (one engine per strip shape, built once, cached).

Modes, per page, same process:
  eager     today: PyTorch FP16, the reference
  compiled  model in channels_last + torch.compile (default mode)
  trt       TensorRT FP16 engine built from an ONNX export of the same weights
Quality: SSIM of each mode vs eager on the raw 2x output (Lanczos half of 4x, as in exp 12/13).
Pages: 10 new random pages (seed 2029), chapters not used in any earlier experiment.

Outputs (experiments/out/exp14_tensorrt/):
  results.jsonl, summary.txt
  compare_pageNN.html   tablet comparer: Eager / TensorRT / Compiled / Difference x8 buttons + slider
  engines/*.plan        cached TensorRT engines (reused on rerun)

Setup once (in the venv):  pip install onnx tensorrt-cu12
Smoke test first (1 page, ~5-10 min incl. engine build):  python experiments/exp14_tensorrt.py --smoke
Full run:                                                 python experiments/exp14_tensorrt.py
Resumable; a STOP file in the out folder stops it cleanly between pages.
Skip a mode if it won't install/build:  --no-trt  or  --no-compile
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, base64, gc, glob, json, random, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import cv2, numpy as np
from common import page_names, read_page, load_sample, detail_scores
import exp06_split_vs_full as e6

OUT = "experiments/out/exp14_tensorrt"
ENG = os.path.join(OUT, "engines")
SCALE = 4


def new_pages(n_per_series=4, seed=2029, limit=10):
    used = {p["cbz"] for p in load_sample()}
    for f in ("experiments/out/exp12_strip_sweep/pages.json", "experiments/out/exp13_strips_vs_routing/pages.json"):
        if os.path.exists(f):
            used |= {p["cbz"] for p in json.load(open(f))}
    rng = random.Random(seed); pages = []
    for series in sorted(os.listdir("Manwhas")):
        cbzs = [c for c in sorted(glob.glob(os.path.join("Manwhas", series, "*.cbz"))) if c not in used]
        for cbz in rng.sample(cbzs, n_per_series):
            pages.append({"series": series, "cbz": cbz, "page": rng.choice(page_names(cbz))})
    rng.shuffle(pages)
    return pages[:limit]


def is_oom(e):
    return "out of memory" in str(e).lower()


# ---------------------------------------------------------------- comparer page
HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 14 page {n}</title><style>
:root{{color-scheme:dark}} body{{margin:0;background:#111;color:#ddd;font:15px system-ui,sans-serif}}
header{{padding:10px 14px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;position:sticky;top:0;background:#111;z-index:3}}
header b{{flex:1 1 100%}} button{{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #555;background:#222;color:#eee}}
button.on{{border-color:#c9a227;color:#f0d878}} .wrap{{position:relative;width:100%;touch-action:pan-y;user-select:none}}
.wrap img{{display:block;width:100%;height:auto}} .over{{position:absolute;inset:0}}
#bar{{position:absolute;top:0;bottom:0;width:2px;background:#c9a227;pointer-events:none}}
p{{padding:0 14px;color:#aaa;line-height:1.5}}
</style></head><body>
<header><b>Page {n} &middot; {info}</b>
<button id="b_e">Eager (today)</button><button id="b_t">TensorRT</button><button id="b_c">Compiled</button>
<button id="b_d">Difference &times;8</button><button id="b_s">Slider</button></header>
<div class="wrap" id="w"><img id="imgBase"><img id="imgOver" class="over"><div id="bar"></div></div>
<p>Slider: left of the gold line is eager (today), right is TensorRT. In "Difference &times;8" black means identical.
SSIM vs eager: TensorRT {s_t}, Compiled {s_c}.</p>
<script>
const SRC={{e:"data:image/jpeg;base64,{e}",t:"data:image/jpeg;base64,{t}",c:"data:image/jpeg;base64,{c}",d:"data:image/jpeg;base64,{d}"}};
const imgBase=document.getElementById("imgBase"),imgOver=document.getElementById("imgOver"),bar=document.getElementById("bar"),w=document.getElementById("w");
let mode="s",pos=50,down=false;
function clip(){{imgOver.style.clipPath="inset(0 "+(100-pos)+"% 0 0)";bar.style.left=pos+"%";}}
function show(m){{mode=m;for(const k of["e","t","c","d","s"])document.getElementById("b_"+k).classList.toggle("on",k===m);
 if(m==="s"){{imgBase.src=SRC.t;imgOver.src=SRC.e;imgOver.style.display="block";bar.style.display="block";clip();}}
 else{{imgBase.src=SRC[m];imgOver.style.display="none";bar.style.display="none";}}}}
function at(ev){{if(mode!=="s")return;const R=w.getBoundingClientRect();pos=Math.max(0,Math.min(100,(ev.clientX-R.left)/R.width*100));clip();}}
w.addEventListener("pointerdown",ev=>{{down=true;at(ev);}});addEventListener("pointerup",()=>{{down=false;}});
w.addEventListener("pointermove",ev=>{{if(down)at(ev);}});
for(const k of["e","t","c","d","s"])document.getElementById("b_"+k).onclick=()=>show(k);
show("s");
</script></body></html>"""


def jpg64(img):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]).decode()


def write_compare(path, n, info, eager2, trt2, comp2, s_t, s_c, src):
    """Most detailed ~900 output rows (450 input rows), full width."""
    h = src.shape[0]
    sc = detail_scores(src, 128).mean(axis=1)
    k = 450 // 128 + 1
    best = int(np.argmax(np.convolve(sc, np.ones(k), "valid"))) if len(sc) > k else 0
    y0 = min(best * 128, max(0, h - 450)) * 2; y1 = min(eager2.shape[0], y0 + 900)
    e = eager2[y0:y1]
    t = trt2[y0:y1] if trt2 is not None else e
    c = comp2[y0:y1] if comp2 is not None else e
    d = cv2.convertScaleAbs(cv2.absdiff(e, t), alpha=8)
    open(path, "w").write(HTML.format(n=n, info=info, e=jpg64(e), t=jpg64(t), c=jpg64(c), d=jpg64(d),
                                      s_t=s_t if s_t is not None else "n/a", s_c=s_c if s_c is not None else "n/a"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-trt", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()

    import torch
    import torch.backends.cudnn as cudnn
    from core.model_loader import load_rrdbnet
    from profiler.cache import load_cache
    from inference.chunker import ChunkMeta, get_strip_height, plan_strips, get_max_chunk_height
    from inference.engine import _preprocess, _pad, _unpad, _postprocess
    from pipeline.reassembler import reassemble
    cudnn.benchmark = True
    model = load_rrdbnet("models/4x-UltraSharp.pth")
    bmap = load_cache()
    sync = torch.cuda.synchronize
    os.makedirs(ENG, exist_ok=True)

    # ---- runners: each takes a padded fp16 NCHW tensor, returns the 4x fp16 NCHW tensor
    def run_eager(x):
        with torch.no_grad():
            return model(x)

    compiled = None
    if not a.no_compile:
        import copy
        m2 = copy.deepcopy(model).to(memory_format=torch.channels_last)
        compiled = torch.compile(m2)

    def run_compiled(x):
        with torch.no_grad():
            return compiled(x.contiguous(memory_format=torch.channels_last)).contiguous()

    trt = None
    if not a.no_trt:
        try:
            import tensorrt as trt
            print("TensorRT", trt.__version__, flush=True)
        except Exception as ex:
            print("TensorRT not importable (pip install onnx tensorrt-cu12), skipping TRT:", ex, flush=True)
            trt = None
    trt_cache = {"shape": None, "engine": None, "ctx": None}
    logger = trt.Logger(trt.Logger.WARNING) if trt else None

    def trt_engine(shape):
        """One engine per padded input shape. Only one is kept live (its activation memory is large)."""
        if trt_cache["shape"] == shape:
            return trt_cache["ctx"]
        trt_cache.update(shape=None, engine=None, ctx=None)  # stream kept; gc.collect(); torch.cuda.empty_cache()
        _, _, H, W = shape
        plan = os.path.join(ENG, f"rrdb_{H}x{W}_fp16_trt{trt.__version__}.plan")
        if not os.path.exists(plan):
            print(f"  building TensorRT engine for {H}x{W} (one-time, a few minutes)...", flush=True)
            onnx_path = os.path.join(ENG, f"rrdb_{H}x{W}_fp16.onnx")
            if not os.path.exists(onnx_path):
                import copy
                mf = copy.deepcopy(model).half().eval()  # FP16 lives in the ONNX file itself (TRT 10+/11 strong typing)
                with torch.no_grad():
                    torch.onnx.export(mf, torch.zeros(shape, device="cuda", dtype=torch.float16), onnx_path, opset_version=17,
                                      input_names=["x"], output_names=["y"], dynamo=False)
                del mf; torch.cuda.empty_cache()
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(open(onnx_path, "rb").read()):
                raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            cfg = builder.create_builder_config()
            cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            blob = builder.build_serialized_network(network, cfg)
            if blob is None:
                raise RuntimeError("TensorRT engine build failed")
            open(plan, "wb").write(bytes(blob))
            del builder, network, parser, cfg, blob
        engine = trt.Runtime(logger).deserialize_cuda_engine(open(plan, "rb").read())
        ctx = engine.create_execution_context()
        trt_cache.update(shape=shape, engine=engine, ctx=ctx)
        return ctx

    def run_trt(x):
        ctx = trt_engine(tuple(x.shape))
        eng = trt_cache["engine"]
        xin = x.half().contiguous()
        n_out = eng.get_tensor_name(1)
        y = torch.empty(tuple(eng.get_tensor_shape(n_out)), dtype=torch.float16 if eng.get_tensor_dtype(n_out) == trt.DataType.HALF else torch.float32, device="cuda")
        ctx.set_tensor_address(eng.get_tensor_name(0), xin.data_ptr())
        ctx.set_tensor_address(n_out, y.data_ptr())
        main = torch.cuda.current_stream()
        if trt_cache.get("stream") is None:
            trt_cache["stream"] = torch.cuda.Stream()  # TRT dislikes the default stream (extra syncs)
        s = trt_cache["stream"]
        s.wait_stream(main)
        if not ctx.execute_async_v3(s.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        main.wait_stream(s)
        return y.half()

    def page(src, runner):
        """Real pipeline planner, batch 1 per strip (TRT engines are batch 1), stitched with the real reassembler."""
        h, w = src.shape[:2]
        H = get_strip_height(w, bmap)
        plan = plan_strips(h, H, max_strip_height=get_max_chunk_height(w, bmap))
        ups = []
        for a0, b0 in plan:
            x = _preprocess(src[a0:b0]); xp, ph, pw = _pad(x, SCALE)
            y = runner(xp)
            ups.append(_postprocess(_unpad(y, ph, pw, SCALE)))
            del x, xp, y
        n = len(plan)
        metas = [ChunkMeta(0, None, k, n, a0, b0, plan[k - 1][1] - a0 if k else 0, b0 - plan[k + 1][0] if k < n - 1 else 0)
                 for k, (a0, b0) in enumerate(plan)]
        out = reassemble(list(zip(ups, metas)), SCALE)
        return out, len(plan), H

    modes = [("eager", run_eager)]
    if compiled is not None: modes.append(("compiled", run_compiled))
    if trt is not None: modes.append(("trt", run_trt))

    pages = new_pages()
    if a.smoke: pages = pages[:1]
    os.makedirs(OUT, exist_ok=True)
    json.dump(pages, open(os.path.join(OUT, "pages.json"), "w"), indent=1)
    path = os.path.join(OUT, "results_smoke.jsonl" if a.smoke else "results.jsonl")
    stop = os.path.join(OUT, "STOP")
    if os.path.exists(stop): os.remove(stop)
    rows = []
    if os.path.exists(path) and not a.smoke:
        for l in open(path):
            try: rows.append(json.loads(l))
            except ValueError: pass
        open(path, "w").writelines(json.dumps(r) + "\n" for r in rows)
    done = {r["i"] for r in rows}
    log = open(path, "w" if a.smoke else "a")
    warmed = set()
    for i, p in enumerate(pages):
        if i in done: continue
        if os.path.exists(stop):
            print("STOP file found: stopping cleanly. Rerun to resume.", flush=True); break
        src = read_page(p["cbz"], p["page"]); h, w = src.shape[:2]
        # warm-up per (mode, width): compile / cuDNN tune / engine build are one-off costs, not timed
        for name, fn in modes:
            if (name, w) not in warmed:
                print(f"  warm-up {name} for width {w}...", flush=True)
                t = time.perf_counter(); page(src, fn); sync()
                print(f"  warm-up {name}: {time.perf_counter() - t:.0f}s", flush=True)
                warmed.add((name, w))
        rec = {"i": i, "series": p["series"], "chapter": os.path.basename(p["cbz"]), "page": p["page"], "size": f"{w}x{h}"}
        outs = {}
        for name, fn in modes:
            gc.collect(); torch.cuda.empty_cache()
            sync(); t = time.perf_counter(); o, n, H = page(src, fn); sync()
            rec[f"t_{name}"] = round(time.perf_counter() - t, 2)
            rec["strips"], rec["strip_h"] = n, H
            outs[name] = e6.half(o); del o
        for name in ("compiled", "trt"):
            if name in outs:
                m = e6.ssim_map(outs["eager"], outs[name])
                rec[f"ssim_{name}"] = round(float(m.mean()), 4)
                rec[f"ssim_min_row_{name}"] = round(float(m.mean(axis=1).min()), 4)
                rec[f"maxdiff_{name}"] = int(cv2.absdiff(outs["eager"], outs[name]).max())
        info = f'{p["series"]} &middot; {os.path.basename(p["cbz"])} &middot; {p["page"]}'
        write_compare(os.path.join(OUT, f"compare_page{i:02d}.html"), i, info, outs["eager"], outs.get("trt"),
                      outs.get("compiled"), rec.get("ssim_trt"), rec.get("ssim_compiled"), src)
        log.write(json.dumps(rec) + "\n"); log.flush(); rows.append(rec)
        line = f'{len(rows):2d}/{len(pages)} page {i:2d} {p["series"][:20]:20s} eager {rec["t_eager"]:5.1f}s'
        for name in ("compiled", "trt"):
            if f"t_{name}" in rec:
                line += f' | {name} {rec["t_" + name]:5.1f}s ssim {rec["ssim_" + name]:.4f}'
        print(line, flush=True)
        del outs, src; gc.collect(); torch.cuda.empty_cache()
    log.close()
    if not rows: return
    base = sum(r["t_eager"] for r in rows)
    lines = [f"{len(rows)} pages (warm-ups excluded)", "eager    : 100%"]
    for name in ("compiled", "trt"):
        if all(f"t_{name}" in r for r in rows):
            lines.append(f"{name:9s}: {100 * sum(r['t_' + name] for r in rows) / base:.0f}% of eager time "
                         f"({base / sum(r['t_' + name] for r in rows):.2f}x faster) | SSIM vs eager mean "
                         f"{st.mean(r['ssim_' + name] for r in rows):.4f}, worst page {min(r['ssim_' + name] for r in rows):.4f}, "
                         f"largest single-pixel difference {max(r['maxdiff_' + name] for r in rows)} levels")
    text = "\n".join(lines); print("\n" + text)
    open(os.path.join(OUT, "summary_smoke.txt" if a.smoke else "summary.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
