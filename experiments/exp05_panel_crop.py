"""Panel-crop estimate: box each connected region of non-solid content, add a margin,
and measure the pixels the network would see vs the whole page. CPU only."""
import os, sys, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np
from common import load_sample, read_page

def content_mask(img, block=16, tol=8):
    """True where a block is not solid colour."""
    h, w = img.shape[:2]
    H, W = -(-h // block), -(-w // block)
    pad = cv2.copyMakeBorder(img, 0, H * block - h, 0, W * block - w, cv2.BORDER_REPLICATE)
    b = pad.reshape(H, block, W, block, 3)
    spread = (b.max(axis=(1, 3)).astype(int) - b.min(axis=(1, 3))).max(-1)
    return (spread > tol).astype(np.uint8)

def crop_area(img, margin):
    h, w = img.shape[:2]
    m = content_mask(img)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))          # join a panel's own fragments
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        boxes.append([max(0, x*16 - margin), max(0, y*16 - margin),
                      min(w, (x+bw)*16 + margin), min(h, (y+bh)*16 + margin)])
    # merge overlapping boxes so no pixel is paid for twice
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j); merged = True; break
            if merged: break
    return sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes), len(boxes)

res, nbox = {}, []
for p in load_sample():
    img = read_page(p["cbz"], p["page"]); h, w = img.shape[:2]
    for margin in (16, 32):
        a, n = crop_area(img, margin)
        res.setdefault((p["series"], margin), []).append(100 * a / (h * w))
        if margin == 16: nbox.append(n)
print(f"Boxes per page: median {st.median(nbox)}, max {max(nbox)}")
for s in sorted({s for s, _ in res}) + ["ALL"]:
    print(s)
    for m in (16, 32):
        v = [x for (ss, mm), xs in res.items() if mm == m and (s == "ALL" or ss == s) for x in xs]
        print(f"  margin {m}px: network sees {st.mean(v):5.1f}% of the page (min {min(v):4.1f}, max {max(v):5.1f})")
