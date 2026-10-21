"""Lab: SVD on one manhwa page.

1. Splits the page (each colour channel) into U * S * V^T and rebuilds it from the top-k singular values,
   for k = 1 ... full, so you can watch the picture come back one "ingredient" at a time.
2. Sid's question: what if we upscale the *compressed* page (the factors U and V) instead of the image?
   Upscales U's rows (height) and V's rows (width) 2x with cubic interpolation, rebuilds, and compares with
   plain bicubic upscaling of the same rank-k page, and with UltraSharp's 2x output of the page.

Output: outputs/lab-svd/<exam>-<page>/index.html (open it in a browser). Local only, gitignored.

  python experiments/lab/svd_page.py                 # exam A, page 005
  python experiments/lab/svd_page.py --exam C --page 012
"""
import argparse, json, os, sys, time, zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from outputs import output_dir  # noqa: E402

KS = [1, 2, 5, 10, 20, 50, 100, 200, 400]
UPSCALE_KS = [20, 100, "full"]


def load_page(exam, page):
    with zipfile.ZipFile(os.path.join("data", "exams", exam, "test.cbz")) as z:
        img = cv2.imdecode(np.frombuffer(z.read(f"{page}.png"), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"could not read page {page} of exam {exam}")
    return img


def svd_channels(img):
    """Per channel: (U, s, Vt) with U h x r, s r, Vt r x w, r = min(h, w)."""
    return [np.linalg.svd(img[:, :, c].astype(np.float32), full_matrices=False) for c in range(3)]


def rebuild(parts, k):
    return np.stack([(U[:, :k] * s[:k]) @ Vt[:k] for U, s, Vt in parts], axis=2)


def upscale_factors(parts, k, h2, w2):
    """2x by interpolating the factors: U (h x k) -> (2h x k), V (w x k) -> (2w x k); S untouched."""
    out = []
    for U, s, Vt in parts:
        U2 = cv2.resize(U[:, :k], (U[:, :k].shape[1], h2), interpolation=cv2.INTER_CUBIC)
        V2 = cv2.resize(np.ascontiguousarray(Vt[:k].T), (k, w2), interpolation=cv2.INTER_CUBIC)
        out.append((U2 * s[:k]) @ V2.T)
    return np.stack(out, axis=2)


def u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam", default="A")
    ap.add_argument("--page", default="005")
    a = ap.parse_args()

    img = load_page(a.exam, a.page)
    h, w = img.shape[:2]
    out = output_dir(os.path.join("outputs", "lab-svd", f"{a.exam}-{a.page}"), exam=a.exam, page=a.page,
                     size=[w, h], note="SVD rebuild + factor-upscale experiment; contains page images, local only")
    print(f"page {a.exam}/{a.page}: {w} x {h}")

    t = time.time()
    parts = svd_channels(img)
    print(f"SVD of 3 channels: {time.time() - t:.1f} s")
    r = len(parts[0][1])

    # 1. singular-value spectrum
    s_all = np.stack([p[1] for p in parts])               # 3 x r
    energy = np.cumsum((s_all ** 2).sum(0)) / (s_all ** 2).sum()
    spectrum = {"s": [[round(float(v), 2) for v in s] for s in s_all], "energy": [round(float(e), 6) for e in energy]}

    # 2. rebuilds
    ks = [k for k in KS if k < r] + [r]
    rebuilds = []
    for k in ks:
        name = f"k{k:04d}.jpg"
        cv2.imwrite(os.path.join(out, name), u8(rebuild(parts, k)), [cv2.IMWRITE_JPEG_QUALITY, 92])
        err = float(np.abs(rebuild(parts, k) - img).mean()) if k <= 400 or k == r else None
        rebuilds.append({"k": k, "file": name, "energy": float(energy[k - 1]),
                         "storage": k * (h + w + 1) / (h * w), "mean_err": err})
        print(f"  k={k:4d}  energy {energy[k-1]*100:7.3f}%  storage {k*(h+w+1)/(h*w)*100:6.1f}% of the pixels  "
              f"mean error {err:.2f} levels")
    cv2.imwrite(os.path.join(out, "original.png"), img)

    # 3. upscale the factors vs bicubic of the same rank-k page
    teacher_path = os.path.join("data", "teacher", "ultrasharp-2x", "exams", a.exam, f"{a.page}.png")
    teacher = cv2.imread(teacher_path, cv2.IMREAD_COLOR) if os.path.exists(teacher_path) else None
    if teacher is not None:
        cv2.imwrite(os.path.join(out, "teacher.png"), teacher)
    ups = []
    for k in UPSCALE_KS:
        kk = r if k == "full" else k
        fac = upscale_factors(parts, kk, 2 * h, 2 * w)
        bic = cv2.resize(rebuild(parts, kk), (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)
        diff = np.abs(fac - bic)
        tag = f"{kk:04d}"
        cv2.imwrite(os.path.join(out, f"up_factors_k{tag}.png"), u8(fac))
        cv2.imwrite(os.path.join(out, f"up_bicubic_k{tag}.png"), u8(bic))
        cv2.imwrite(os.path.join(out, f"up_diff_k{tag}.png"), u8(np.minimum(diff * 50, 255)))
        row = {"k": kk, "label": str(k), "factors": f"up_factors_k{tag}.png", "bicubic": f"up_bicubic_k{tag}.png",
               "diff": f"up_diff_k{tag}.png", "max_diff_factors_vs_bicubic": float(diff.max()),
               "mean_diff_factors_vs_bicubic": float(diff.mean())}
        if teacher is not None and teacher.shape == fac.shape:
            row["mean_diff_vs_teacher"] = float(np.abs(u8(fac).astype(np.float32) - teacher).mean())
            row["mean_diff_bicubic_vs_teacher"] = float(np.abs(u8(bic).astype(np.float32) - teacher).mean())
        ups.append(row)
        print(f"  upscale k={k}: factors vs bicubic max diff {diff.max():.4f}, mean {diff.mean():.5f} levels"
              + (f" | vs UltraSharp: factors {row['mean_diff_vs_teacher']:.2f}, bicubic "
                 f"{row['mean_diff_bicubic_vs_teacher']:.2f}" if "mean_diff_vs_teacher" in row else ""))

    data = {"exam": a.exam, "page": a.page, "w": w, "h": h, "rank": r, "rebuilds": rebuilds, "spectrum": spectrum,
            "upscales": ups, "teacher": "teacher.png" if teacher is not None else None}
    json.dump(data, open(os.path.join(out, "results.json"), "w"), indent=1)
    open(os.path.join(out, "index.html"), "w").write(HTML.replace("__DATA__", json.dumps(data)))
    print(f"\nopen {os.path.join(out, 'index.html')}")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>SVD lab</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{background:#111;color:#ddd;font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px}
h1,h2{font-weight:600}h2{margin-top:40px}.row{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
input[type=range]{width:420px}.stat{font-variant-numeric:tabular-nums;color:#9cf}
.view{overflow:auto;max-height:80vh;border:1px solid #333;margin-top:10px}.view img{display:block;image-rendering:auto}
button{background:#222;color:#ddd;border:1px solid #444;padding:6px 12px;border-radius:6px;cursor:pointer}
button.on{background:#2a4a7a;border-color:#5a8ad0}canvas{background:#181818;border:1px solid #333}
.note{color:#999;max-width:900px}
</style></head><body>
<h1>SVD on one page</h1>
<p class="note">Every colour channel is split into U &middot; S &middot; V<sup>T</sup>. Each singular value is one
"ingredient": a column profile times a row profile, weighted by how much it matters. Slide k to rebuild the page
from only the top k ingredients.</p>
<div class="row"><b>k</b><input id="k" type="range" min="0" step="1"><span class="stat" id="kstat"></span>
<label><input id="zoom" type="checkbox"> 2&times; zoom</label></div>
<div class="view"><img id="kimg"></div>
<h2>Spectrum</h2><p class="note">Singular values on a log scale (blue/green/red = B/G/R channels), and the share of
the page's energy kept by the top k (white).</p><canvas id="spec" width="900" height="300"></canvas>
<h2>Upscale the factors instead of the image</h2>
<p class="note">Interpolate U's rows to twice the height and V's rows to twice the width, keep S, rebuild. Compare
with plain bicubic upscaling of the same rank-k page, and with UltraSharp. "Diff" shows |factors &minus; bicubic|
&times; 50.</p>
<div class="row" id="upk"></div><div class="row" id="upsrc" style="margin-top:8px"></div>
<div class="row stat" id="upstat"></div><div class="view"><img id="upimg"></div>
<script>
const D=__DATA__;const R=D.rebuilds;const k=document.getElementById('k');k.max=R.length-1;k.value=4;
const zoom=document.getElementById('zoom');
function showK(){const r=R[k.value];const im=document.getElementById('kimg');im.src=r.file;
 im.style.width=(zoom.checked?2*D.w:D.w)+'px';
 document.getElementById('kstat').textContent=`k = ${r.k} of ${D.rank}  ·  energy ${(r.energy*100).toFixed(3)}%  ·  `+
 `storage ${(r.storage*100).toFixed(1)}% of the pixels  ·  mean error ${r.mean_err.toFixed(2)} levels`;}
k.oninput=showK;zoom.onchange=()=>{showK();showUp();};showK();
// spectrum
const c=document.getElementById('spec').getContext('2d');const W=900,H=300,n=D.spectrum.s[0].length;
const lmax=Math.log10(Math.max(...D.spectrum.s.map(a=>a[0]))),lmin=Math.log10(Math.max(1e-2,Math.min(...D.spectrum.s.map(a=>a[a.length-1]))));
['#5af','#5d5','#f66'].forEach((col,ch)=>{c.strokeStyle=col;c.beginPath();D.spectrum.s[ch].forEach((v,i)=>{
 const x=40+i/(n-1)*(W-60),y=H-20-(Math.log10(Math.max(v,1e-2))-lmin)/(lmax-lmin)*(H-40);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();});
c.strokeStyle='#eee';c.beginPath();D.spectrum.energy.forEach((e,i)=>{const x=40+i/(n-1)*(W-60),y=H-20-e*(H-40);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();
c.fillStyle='#888';c.fillText('k = 1',40,H-5);c.fillText('k = '+n,W-60,H-5);
// upscale viewer
let ui=0,src='factors';const srcs=['factors','bicubic','diff'].concat(D.teacher?['teacher']:[]);
function btns(id,items,cur,set){const el=document.getElementById(id);el.innerHTML='';items.forEach((it,i)=>{
 const b=document.createElement('button');b.textContent=it.t;if(it.v===cur)b.className='on';b.onclick=()=>{set(it.v);draw();};el.appendChild(b);});}
function draw(){btns('upk',D.upscales.map((u,i)=>({t:'k = '+u.label,v:i})),ui,v=>ui=v);
 btns('upsrc',srcs.map(s=>({t:s,v:s})),src,v=>src=v);showUp();}
function showUp(){const u=D.upscales[ui];const im=document.getElementById('upimg');im.src=src==='teacher'?D.teacher:u[src];
 im.style.width=(zoom.checked?4*D.w:2*D.w)+'px';
 let t=`factors vs bicubic: max ${u.max_diff_factors_vs_bicubic.toFixed(4)}, mean ${u.mean_diff_factors_vs_bicubic.toFixed(5)} levels`;
 if(u.mean_diff_vs_teacher!==undefined)t+=`  ·  mean distance to UltraSharp: factors ${u.mean_diff_vs_teacher.toFixed(2)}, bicubic ${u.mean_diff_bicubic_vs_teacher.toFixed(2)}`;
 document.getElementById('upstat').textContent=t;}
draw();
</script></body></html>"""

if __name__ == "__main__":
    main()
