"""Lab: is HDFyer's model secretly low-rank? (SVD on the weights, not the image.)

For every convolution the folded model actually runs, reshapes its weights to a matrix (out x in*kh*kw), takes the
SVD and asks: how many "ingredients" does this layer really use? A layer whose singular values fall off a cliff can be
split into two thin layers (k filters, then a 1x1 mix) that do less arithmetic per pixel for nearly the same output.

Then the go/no-go test: truncate every layer at an energy threshold (99%, 99.9%, ...), run the model on real crops
from two exam pages, and measure (a) arithmetic saved and (b) how far the output moves from the untouched model.
Truncation here only simulates the split (same weight shapes), so it measures quality, not speed.

  python experiments/lab/svd_weights.py
Output: printed tables + outputs/lab-svd-weights/results.json
"""
import json, os, sys, zipfile

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from hdfyer.digest.model import fold_reparam  # noqa: E402
from outputs import output_dir  # noqa: E402

MODEL = os.path.join("models", "hdfyer", "hdfyer-2x-s-r2.pth")
THRESHOLDS = [0.90, 0.95, 0.99, 0.999]
PAGES = [("A", "005"), ("D", "005")]   # soft painted page and gritty texture page
CROP, CROPS_PER_PAGE = 256, 3


def load_model(dev):
    from spandrel import ModelLoader
    d = ModelLoader().load_from_file(MODEL)
    d.to(dev).eval()
    fold_reparam(d.model)
    return d.model


def used_convs(model, x):
    """Conv2d modules that actually run in the folded model's forward pass (in call order)."""
    seen, hooks = [], []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: seen.append(n) if n not in seen else None))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    mods = dict(model.named_modules())
    return [(n, mods[n]) for n in seen]


def split_cost(o, i, kh, kw, r):
    """Multiply-adds per pixel: original conv vs (r filters of i x kh x kw) + (1x1, r -> o)."""
    return o * i * kh * kw, r * (i * kh * kw + o)


def crops(dev):
    out = []
    for exam, page in PAGES:
        with zipfile.ZipFile(os.path.join("data", "exams", exam, "test.cbz")) as z:
            img = cv2.imdecode(np.frombuffer(z.read(f"{page}.png"), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        rng = np.random.default_rng(7)
        for _ in range(CROPS_PER_PAGE):
            y, x = int(rng.integers(0, h - CROP)), int(rng.integers(0, w - CROP))
            c = img[y:y + CROP, x:x + CROP, ::-1].astype(np.float32) / 255
            out.append(torch.from_numpy(c.copy()).permute(2, 0, 1)[None].to(dev))
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = output_dir("lab-svd-weights", model=MODEL)
    model = load_model(dev)
    xs = crops(dev)
    convs = used_convs(model, xs[0])

    layers, total_full = [], 0
    print(f"{'layer':38s} {'shape':>16s} {'rank':>4s} {'r@99%':>6s} {'r@99.9%':>7s} {'breakeven r':>11s}")
    for name, m in convs:
        W = m.weight.detach().float().cpu()
        o, i, kh, kw = W.shape
        M = W.reshape(o, -1).numpy()
        s = np.linalg.svd(M, compute_uv=False)
        e = np.cumsum(s ** 2) / (s ** 2).sum()
        full, _ = split_cost(o, i, kh, kw, 1)
        breakeven = full / (i * kh * kw + o)
        rk = {t: int(np.searchsorted(e, t) + 1) for t in THRESHOLDS}
        layers.append({"name": name, "shape": [o, i, kh, kw], "rank": len(s), "s": [float(v) for v in s],
                       "rank_at": {str(t): r for t, r in rk.items()}, "breakeven": breakeven, "cost": full})
        total_full += full
        print(f"{name:38s} {str((o, i, kh, kw)):>16s} {len(s):4d} {rk[0.99]:6d} {rk[0.999]:7d} {breakeven:11.1f}")

    with torch.no_grad():
        refs = [model(x).clamp(0, 1) for x in xs]
    originals = {name: m.weight.detach().clone() for name, m in convs}

    print(f"\n{'threshold':>9s} {'arithmetic vs today':>20s} {'mean diff':>10s} {'p99 diff':>9s} {'max diff':>9s}  (levels, 0-255)")
    trials = []
    for t in THRESHOLDS:
        cost = 0
        for (name, m), L in zip(convs, layers):
            o, i, kh, kw = L["shape"]
            r = L["rank_at"][str(t)]
            full, split = split_cost(o, i, kh, kw, r)
            if split < full:   # only split where it saves work
                W = originals[name].float()
                U, S, Vt = torch.linalg.svd(W.reshape(o, -1), full_matrices=False)
                m.weight.data = ((U[:, :r] * S[:r]) @ Vt[:r]).reshape(o, i, kh, kw).to(m.weight.dtype)
                cost += split
            else:
                m.weight.data = originals[name].clone()
                cost += full
        with torch.no_grad():
            d = torch.cat([((model(x).clamp(0, 1) - ref).abs() * 255).flatten() for x, ref in zip(xs, refs)])
        row = {"threshold": t, "arithmetic_pct": 100 * cost / total_full, "mean": float(d.mean()),
               "p99": float(torch.quantile(d[:: max(1, d.numel() // 1_000_000)], 0.99)), "max": float(d.max())}
        trials.append(row)
        print(f"{t*100:8.1f}% {row['arithmetic_pct']:19.1f}% {row['mean']:10.3f} {row['p99']:9.2f} {row['max']:9.1f}")
    for name, m in convs:
        m.weight.data = originals[name]

    json.dump({"model": MODEL, "device": dev, "layers": layers, "trials": trials,
               "crops": {"pages": PAGES, "size": CROP, "per_page": CROPS_PER_PAGE}},
              open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    print(f"\nSaved {os.path.join(out_dir, 'results.json')}")
    print("Reading it: a layer is worth splitting only if r@99.9% is well below its breakeven r. For a real speed-up the "
          "whole model needs a big arithmetic cut at a mean diff well under 1 level, and even then the model is partly "
          "memory-bound, so time saved < arithmetic saved.")


if __name__ == "__main__":
    main()
