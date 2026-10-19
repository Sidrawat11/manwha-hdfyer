"""Turn exp 13's seam images ([today | strips | difference x8], 1600 px each, 12 px gaps) into
full-width tablet comparers: one image at a time, buttons to switch, slider between today and strips."""
import base64, glob, os, re
import cv2

OUT = "outputs/exp13_strips_vs_routing"
TPL = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seam check page {n}</title><style>
:root{{color-scheme:dark}} body{{margin:0;background:#111;color:#ddd;font:15px system-ui,sans-serif}}
header{{padding:10px 14px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;position:sticky;top:0;background:#111;z-index:3}}
header b{{flex:1 1 100%}} button{{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #555;background:#222;color:#eee}}
button.on{{border-color:#c9a227;color:#f0d878}} .wrap{{position:relative;width:100%;touch-action:pan-y;user-select:none}}
.wrap img{{display:block;width:100%;height:auto}} .top{{position:absolute;inset:0}}
#bar{{position:absolute;top:0;bottom:0;width:2px;background:#c9a227;pointer-events:none}}
p{{padding:0 14px;color:#aaa;line-height:1.5}}
</style></head><body>
<header><b>Seam check, page {n}: the band of rows that scored lowest (2&times; output)</b>
<button id="bt">Today</button><button id="bs">Smarter strips</button><button id="bd">Difference &times;8</button><button id="bl">Slider</button></header>
<div class="wrap" id="w"><img id="base" src="data:image/jpeg;base64,{s}"><img id="top" class="top" src="data:image/jpeg;base64,{t}"><div id="bar"></div></div>
<p>Today and Smarter strips should look identical. In "Difference &times;8", black means identical; a seam would show as a straight horizontal line across the whole width.</p>
<script>
const T="data:image/jpeg;base64,{t}",S="data:image/jpeg;base64,{s}",D="data:image/jpeg;base64,{d}";
const baseImg=document.getElementById("base"),topImg=document.getElementById("top"),bar=document.getElementById('bar'),w=document.getElementById('w');
let mode='slider',pos=50,down=false;
function show(m){{mode=m;for(const[id,k]of[['bt','t'],['bs','s'],['bd','d'],['bl','slider']])document.getElementById(id).classList.toggle('on',k===m);
 if(m==='slider'){{baseImg.src=S;topImg.src=T;topImg.style.display='block';bar.style.display='block';clip();}}
 else{{baseImg.src=m==='t'?T:m==='s'?S:D;topImg.style.display='none';bar.style.display='none';}}}}
function clip(){{topImg.style.clipPath='inset(0 '+(100-pos)+'% 0 0)';bar.style.left=pos+'%';}}
function at(e){{if(mode!=='slider')return;const R=w.getBoundingClientRect();pos=Math.max(0,Math.min(100,(e.clientX-R.left)/R.width*100));clip();}}
w.addEventListener('pointerdown',e=>{{down=true;at(e)}});addEventListener('pointerup',()=>down=false);w.addEventListener('pointermove',e=>{{if(down)at(e)}});
bt.onclick=()=>show('t');bs.onclick=()=>show('s');bd.onclick=()=>show('d');bl.onclick=()=>show('slider');show('slider');
</script></body></html>"""
for path in sorted(glob.glob(os.path.join(OUT, "seam_page*.jpg"))):
    n = int(re.search(r"seam_page(\d+)", path).group(1))
    img = cv2.imread(path); W = (img.shape[1] - 24) // 3
    parts = [img[:, 0:W], img[:, W + 12:2 * W + 12], img[:, 2 * W + 24:3 * W + 24]]
    enc = lambda a: base64.b64encode(cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]).decode()
    out = os.path.join(OUT, f"seam_compare_page{n:02d}.html")
    open(out, "w").write(TPL.format(n=n, t=enc(parts[0]), s=enc(parts[1]), d=enc(parts[2])))
    print(out, parts[0].shape)
