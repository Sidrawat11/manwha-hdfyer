"""Tapestry experiment: split a manhwa page into line art + everything else,
enlarge each 2x separately, recombine, and measure what was lost.
CPU only. Does not touch the pipeline.

Usage:
  python experiments/exp01_tapestry.py --random "Manwhas/<series folder>" --out experiments/out
  python experiments/exp01_tapestry.py --image page.jpg --out experiments/out
"""
import argparse, glob, os, random, zipfile, json
import cv2, numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

def pick_random(series_dir, seed):
    rng = random.Random(seed)
    cbz = rng.choice(sorted(glob.glob(os.path.join(series_dir, "*.cbz"))))
    with zipfile.ZipFile(cbz) as z:
        names = sorted(n for n in z.namelist()
                       if n.lower().endswith(IMG_EXT) and "__MACOSX" not in n
                       and not os.path.basename(n).startswith("."))
        name = rng.choice(names)
        img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
    return img, cbz, name, len(names)

def line_mask(src, kernel, thresh):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    strokes = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)   # dark strokes thinner than kernel
    return cv2.dilate((strokes > thresh).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))

def census(src, mask, tile=64, solid_spread=8):
    h, w = src.shape[:2]
    counts = {"solid": 0, "smooth": 0, "edge": 0}
    overlay = src.copy()
    tints = {"smooth": (255, 120, 0), "edge": (0, 0, 255)}  # BGR: blue, red
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            t = src[y:y+tile, x:x+tile]
            spread = (t.reshape(-1, 3).max(0).astype(int) - t.reshape(-1, 3).min(0)).max()
            if spread <= solid_spread:
                kind = "solid"
            elif mask[y:y+tile, x:x+tile].any():
                kind = "edge"
            else:
                kind = "smooth"
            counts[kind] += 1
            if kind in tints:
                tinted = np.full_like(t, tints[kind])
                overlay[y:y+tile, x:x+tile] = cv2.addWeighted(t, 0.6, tinted, 0.4, 0)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}, overlay

def main():
    ap = argparse.ArgumentParser()
    src_g = ap.add_mutually_exclusive_group(required=True)
    src_g.add_argument("--image")
    src_g.add_argument("--random", help="series folder of CBZs")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--thresh", type=int, default=40)
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else random.randrange(1_000_000)
    if a.random:
        src, cbz, name, n = pick_random(a.random, seed)
        label = f"{os.path.basename(cbz)} :: {name} (of {n} pages)"
    else:
        src, label = cv2.imread(a.image), a.image
    h, w = src.shape[:2]

    mask = line_mask(src, a.kernel, a.thresh)
    tapestry = cv2.inpaint(src, mask, 3, cv2.INPAINT_TELEA)

    up = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    alpha = up(mask).astype(np.float32)[..., None] / 255
    plain = up(src)
    rebuilt = (alpha * plain + (1 - alpha) * up(tapestry)).round().astype(np.uint8)

    shares, overlay = census(src, mask)
    psnr = cv2.PSNR(rebuilt, plain)
    stats = {"page": label, "seed": seed, "size": f"{w}x{h}",
             "line_pct": round(100 * mask.mean() / 255, 1),
             **{f"{k}_tiles_pct": round(100 * v, 1) for k, v in shares.items()},
             "psnr_db": round(psnr, 1)}

    os.makedirs(a.out, exist_ok=True)
    v = slice(0, min(h, 1600))
    cv2.imwrite(os.path.join(a.out, "tapestry.png"),
                np.hstack([src[v], cv2.cvtColor(mask[v], cv2.COLOR_GRAY2BGR), tapestry[v]]))
    cv2.imwrite(os.path.join(a.out, "rebuilt_2x.png"), rebuilt)
    cv2.imwrite(os.path.join(a.out, "census.png"), overlay)
    cv2.imwrite(os.path.join(a.out, "original.png"), src)
    with open(os.path.join(a.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
