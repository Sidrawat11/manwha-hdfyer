"""Shared helpers for the routing experiments (CPU only, no torch)."""
import glob, json, os, random, zipfile
import cv2, numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
SAMPLE_PATH = "experiments/out/sample.json"

def page_names(cbz):
    with zipfile.ZipFile(cbz) as z:
        return sorted(n for n in z.namelist()
                      if n.lower().endswith(IMG_EXT) and "__MACOSX" not in n
                      and not os.path.basename(n).startswith("."))

def read_page(cbz, name):
    with zipfile.ZipFile(cbz) as z:
        return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)

def build_sample(root="Manwhas", per_series=20, seed=2026):
    """Random chapter first, then a random page in it; no chapter used twice per series."""
    rng = random.Random(seed)
    sample = []
    for series in sorted(os.listdir(root)):
        cbzs = sorted(glob.glob(os.path.join(root, series, "*.cbz")))
        for cbz in rng.sample(cbzs, min(per_series, len(cbzs))):
            names = page_names(cbz)
            sample.append({"series": series, "cbz": cbz, "page": rng.choice(names)})
    os.makedirs(os.path.dirname(SAMPLE_PATH), exist_ok=True)
    with open(SAMPLE_PATH, "w") as f:
        json.dump({"seed": seed, "pages": sample}, f, indent=1)
    return sample

def load_sample():
    with open(SAMPLE_PATH) as f:
        return json.load(f)["pages"]

def line_mask(src, kernel=7, thresh=40):
    """Thin strokes, dark-on-light (black-hat) and light-on-dark (top-hat)."""
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) > thresh
    light = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k) > thresh
    m = ((dark | light).astype(np.uint8)) * 255
    return cv2.dilate(m, np.ones((3, 3), np.uint8))

def classify_tiles(src, mask, tile, solid_spread=8, min_line_px=None):
    """Returns a (rows, cols) array: 0 solid, 1 smooth, 2 edge."""
    h, w = src.shape[:2]
    if min_line_px is None:
        min_line_px = max(1, (tile * tile) // 200)   # 0.5% of the tile
    rows, cols = -(-h // tile), -(-w // tile)
    kinds = np.zeros((rows, cols), np.uint8)
    for r in range(rows):
        for c in range(cols):
            ys, xs = slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile)
            t = src[ys, xs].reshape(-1, 3)
            if int((t.max(0).astype(int) - t.min(0)).max()) <= solid_spread:
                kinds[r, c] = 0
            elif int((mask[ys, xs] > 0).sum()) >= min_line_px:
                kinds[r, c] = 2
            else:
                kinds[r, c] = 1
    return kinds

def detail_scores(src, tile):
    """Per-square fine-detail strength relative to brightness (experiment 8).
    Detail = image minus a blurred copy; score = its spread / (mean brightness + 20)."""
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    h, w = gray.shape
    R, C = -(-h // tile), -(-w // tile)
    s = np.zeros((R, C), np.float32)
    for r in range(R):
        for c in range(C):
            ys, xs = slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile)
            s[r, c] = detail[ys, xs].std() / (gray[ys, xs].mean() + 20.0)
    return s

def classify_detail(src, tile, thresh=0.02):
    """Line rule OR detail rule. Solid squares stay solid."""
    kinds = classify_tiles(src, line_mask(src), tile)
    kinds[(kinds == 1) & (detail_scores(src, tile) > thresh)] = 2
    return kinds
