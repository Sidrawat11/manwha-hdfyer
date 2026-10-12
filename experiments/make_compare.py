"""Build tablet-friendly before/after comparers from exp 12's close-ups.

Each showcase/pageNN_closeup.jpg is [reader-app 2x | 16 px gap | upscaler 2x], 1600 px wide each.
This writes showcase/compare_pageNN.html: both images stacked at full screen width; drag anywhere
(or use the slider) to move the divider, tap the button to flip between them.
Self-contained (images embedded), opens in any browser. Rerun any time to pick up new pages.
"""
import base64, glob, os, re
import cv2

SHOW = "experiments/out/exp12_strip_sweep/showcase"
TPL = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Page {n}: reader app vs upscaler</title>
<style>
 :root{{color-scheme:dark}} body{{margin:0;background:#111;color:#ddd;font:15px system-ui,sans-serif}}
 header{{display:flex;gap:12px;align-items:center;padding:10px 14px;position:sticky;top:0;background:#111;z-index:3}}
 header b{{flex:1}} button{{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #555;background:#222;color:#eee}}
 .wrap{{position:relative;width:100%;touch-action:pan-y;user-select:none}}
 .wrap img{{display:block;width:100%;height:auto}}
 #before{{position:absolute;inset:0}}
 #bar{{position:absolute;top:0;bottom:0;width:2px;background:#c9a227;pointer-events:none}}
 .tag{{position:absolute;top:8px;padding:4px 8px;border-radius:6px;background:#000a;font-size:13px}}
 input[type=range]{{width:100%;margin:10px 0}}
</style></head><body>
<header><b>Page {n} &middot; left of the line: reader app &middot; right: upscaler</b><button id="flip">Show upscaler</button></header>
<div class="wrap" id="wrap">
 <img id="after" src="data:image/jpeg;base64,{after}" alt="upscaler">
 <img id="before" src="data:image/jpeg;base64,{before}" alt="reader app">
 <div id="bar"></div>
 <span class="tag" style="left:8px">Reader app</span><span class="tag" style="right:8px">Upscaler</span>
</div>
<input type="range" id="r" min="0" max="100" value="50" aria-label="Divider position">
<script>
 const b=document.getElementById('before'),bar=document.getElementById('bar'),r=document.getElementById('r'),
       w=document.getElementById('wrap'),f=document.getElementById('flip');
 function set(p){{p=Math.max(0,Math.min(100,p));r.value=p;b.style.clipPath='inset(0 '+(100-p)+'% 0 0)';bar.style.left=p+'%';
   f.textContent=p>50?'Show upscaler':'Show reader app';}}
 function at(e){{const x=(e.touches?e.touches[0].clientX:e.clientX),R=w.getBoundingClientRect();set((x-R.left)/R.width*100);}}
 let down=false; w.addEventListener('pointerdown',e=>{{down=true;at(e)}}); addEventListener('pointerup',()=>down=false);
 w.addEventListener('pointermove',e=>{{if(down)at(e)}}); r.addEventListener('input',()=>set(+r.value));
 f.addEventListener('click',()=>set(+r.value>50?0:100)); set(50);
</script></body></html>"""

made = []
for path in sorted(glob.glob(os.path.join(SHOW, "page*_closeup.jpg"))):
    n = re.search(r"page(\d+)_", path).group(1)
    img = cv2.imread(path); half = (img.shape[1] - 16) // 2
    enc = lambda a: base64.b64encode(cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]).decode()
    out = os.path.join(SHOW, f"compare_page{n}.html")
    open(out, "w").write(TPL.format(n=int(n), before=enc(img[:, :half]), after=enc(img[:, half + 16:])))
    made.append(out)
print("\n".join(made))
