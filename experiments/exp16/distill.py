"""Experiment 16: distillation. Teach SPAN (small, ~20x faster) to reproduce today's 4x-UltraSharp look.

Teacher = 4x-UltraSharp, run in 512-row strips, stitched, Lanczos-halved to 2x: today's pipeline output
          BEFORE sharpening (sharpening stays a separate, tunable step after the model).
Student = SPAN S 2x, starting from IllustrationJaNai V3detail SPAN weights (it already knows how to
          upscale; we retrain its taste).
Data    = <data>/train_*.cbz and <data>/test.cbz (30 pages, held-out chapters); the current set is data/training/v3
          (548 pages from 3 series),
          built by experiments/exp16/make_dataset.py. No chapter is in both.

Stages (each resumable; `all` runs them in order):
  teacher  run the teacher on every train/test page, save the 2x answer as lossless PNG
  bank     cut training patches: 112x112 input / 224x224 answer, detail-weighted
           (80% detailed, 15% flat, 5% blank: gutters and bubbles still need to stay clean).
           Plus a fixed validation bank from the TEST pages, labelled blank / flat / detail.
  train    L1 loss vs the teacher, AdamW + cosine schedule, bf16 autocast, random 96px crops,
           flips and rotations. Validation every --val-every steps, per category. Keeps best + last.
  eval     whole test pages: student vs teacher (similarity, mean/max difference, seconds per page),
           the starting JaNai SPAN for reference, per-category similarity, and slider comparers.
  exam     teacher + eval on an UNSEEN series (no training files needed), for any checkpoint:
             python experiments/exp16/make_dataset.py --exam "<series folder>"
             python experiments/exp16/distill.py exam --data data/exams/A
                 --out outputs/exp16-exam/A_run2
                 --student models/hdfyer/hdfyer-2x-s-r2.pth

Self-contained on purpose (torch, spandrel, opencv, numpy, tqdm only) so it runs on a rented GPU
without the rest of the pipeline. Comparers reuse the Exp 15 page writer.

Local smoke test (few minutes, proves the loop works and the loss goes down):
  python experiments/exp16/distill.py all --smoke --out outputs/exp16-smoke
Full run (cloud: see experiments/exp16/cloud_run.py):
  python experiments/exp16/distill.py all --out runs/exp16
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, glob, io, json, math, random, sys, time, zipfile
import cv2, numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # experiments/ (for the Exp 15 comparer)

STRIP, OVERLAP = 512, 32
DEV = os.environ.get("EXP16_DEVICE", "cuda")  # "cpu" only for testing the plumbing


def sync():
    import torch
    if DEV == "cuda":
        torch.cuda.synchronize()
P, J = 96, 8            # training crop (input px) and jitter margin; stored patches are P + 2J
PS = P + 2 * J          # 112 input, 224 answer
CATS = ("blank", "flat", "detail")


# ---------------------------------------------------------------- data helpers
def read_cbz(path):
    out = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    out.append((n, img))
    return out


def split_files(data, need_train=True):
    train = sorted(glob.glob(os.path.join(data, "train_*.cbz"))) if need_train else []
    test = os.path.join(data, "test.cbz")
    if (need_train and not train) or not os.path.exists(test):
        sys.exit(f"Expected {data}/train_*.cbz and {data}/test.cbz (run experiments/exp16/make_dataset.py)")
    return train, test


def plan_strips(h, H, ov=OVERLAP):
    """Evenly spaced fixed-height strips with ~ov rows of overlap (same idea as inference.chunker.plan_strips)."""
    if h <= H:
        return [(0, h)]
    n = math.ceil((h - H) / (H - ov)) + 1
    build = lambda n: [(round(k * (h - H) / (n - 1)), round(k * (h - H) / (n - 1)) + H) for k in range(n)]
    fits = lambda p: all(p[k - 1][1] - p[k][0] + p[k][1] - p[k + 1][0] <= H for k in range(1, len(p) - 1))
    plan = build(n)
    while not fits(plan) and n > 2:
        n -= 1; plan = build(n)
    if len(plan) == 2 and plan[0][1] - plan[1][0] < ov:
        t = math.ceil((h + ov) / 2); plan = [(0, t), (h - t, h)]
    return plan


def stitch(ups, plan, s):
    """Linear blend across each overlap (same as pipeline.reassembler)."""
    pieces = []
    for k, (a, b) in enumerate(plan):
        top = (plan[k - 1][1] - a) * s if k else 0
        bot = (b - plan[k + 1][0]) * s if k < len(plan) - 1 else 0
        up = ups[k]
        if top:
            alpha = np.linspace(0, 1, top)[:, None, None]
            pieces.append((ups[k - 1][-top:] * (1 - alpha) + up[:top] * alpha).astype(np.uint8))
        pieces.append(up[top:up.shape[0] - bot])
    out = np.concatenate(pieces, axis=0)
    assert out.shape[0] == (plan[-1][1] - plan[0][0]) * s, "stitch height mismatch"
    return out


def read_names(path):
    with zipfile.ZipFile(path) as z:
        return [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]


def tile_categories(img, t=32):
    """Per t x t input tile: 0 blank (spread <= 8), 1 flat (detail < 0.02), 2 detail. Same measures as Exp 2/8."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    H, W = g.shape[0] // t, g.shape[1] // t
    gt = g[:H * t, :W * t].reshape(H, t, W, t)
    ht = hp[:H * t, :W * t].reshape(H, t, W, t)
    spread = gt.max(axis=(1, 3)) - gt.min(axis=(1, 3))
    detail = ht.std(axis=(1, 3)) / (gt.mean(axis=(1, 3)) + 20)
    cat = np.where(spread <= 8, 0, np.where(detail < 0.02, 1, 2)).astype(np.uint8)
    return cat, detail


# ---------------------------------------------------------------- models
def load_model(path, half=True):
    import torch
    from spandrel import ModelLoader
    d = ModelLoader().load_from_file(path)
    d.to(DEV).eval()
    return d.half() if half and DEV == "cuda" else d.float()


def to_t(img):
    import torch
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(DEV).to(torch.float16 if DEV == "cuda" else torch.float32).div_(255)


def from_t(y):
    y = (y.squeeze(0).float().clamp_(0, 1) * 255).round_().byte().cpu().numpy().transpose(1, 2, 0)
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def run_model(desc, img, strip=STRIP):
    """Whole page through a spandrel model in strips; returns the model-scale output."""
    import torch
    h = img.shape[0]
    plan = plan_strips(h, min(strip, h))
    ups = []
    with torch.inference_mode():
        for a, b in plan:
            ups.append(from_t(desc(to_t(img[a:b]))))
    return stitch(ups, plan, desc.scale)


def half(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------- stage: teacher
def stage_teacher(a):
    import torch
    torch.backends.cudnn.benchmark = DEV == "cuda"
    train, test = split_files(a.data, a.stage != "exam")
    tdir = os.path.join(a.out, "teacher"); os.makedirs(tdir, exist_ok=True)
    times_path = os.path.join(tdir, "times.json")
    times = json.load(open(times_path)) if os.path.exists(times_path) else {}
    teacher = None
    files = train + [test]
    for f in files:
        pages = read_cbz(f)[: a.pages_per_cbz] if a.pages_per_cbz else read_cbz(f)
        sub = os.path.join(tdir, os.path.splitext(os.path.basename(f))[0]); os.makedirs(sub, exist_ok=True)
        for name, img in tqdm(pages, desc=f"teacher {os.path.basename(f)}", unit="page"):
            dst = os.path.join(sub, os.path.splitext(name)[0] + ".png")
            if os.path.exists(dst):
                continue
            if teacher is None:
                teacher = load_model(a.teacher)
                assert teacher.scale == 4, "teacher should be the 4x model"
                run_model(teacher, img[:256])  # warm-up
            sync(); t = time.perf_counter()
            out2 = half(run_model(teacher, img))
            sync(); times[f"{os.path.basename(f)}/{name}"] = round(time.perf_counter() - t, 3)
            assert out2.shape[:2] == (img.shape[0] * 2, img.shape[1] * 2), f"teacher output size on {name}"
            cv2.imwrite(dst + ".tmp.png", out2, [cv2.IMWRITE_PNG_COMPRESSION, 1]); os.replace(dst + ".tmp.png", dst)
            json.dump(times, open(times_path, "w"))
    print("teacher: done", flush=True)


def teacher_img(a, cbz, name):
    p = os.path.join(a.out, "teacher", os.path.splitext(os.path.basename(cbz))[0], os.path.splitext(name)[0] + ".png")
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"missing teacher output {p}: run the teacher stage first")
    return img


# ---------------------------------------------------------------- stage: bank
def sample_positions(img, n, rng, mix, stride=24):
    """Top-left corners of PS x PS patches, mixed blank/flat/detail; detail weighted by its score."""
    h, w = img.shape[:2]
    if h < PS or w < PS:
        return []
    cat, det = tile_categories(img, 32)
    ys = list(range(0, h - PS + 1, stride)); xs = list(range(0, w - PS + 1, stride))
    cand = {0: [], 1: [], 2: []}; wts = []
    for y in ys:
        for x in xs:
            c = cat[min((y + PS // 2) // 32, cat.shape[0] - 1), min((x + PS // 2) // 32, cat.shape[1] - 1)]
            d = det[min((y + PS // 2) // 32, det.shape[0] - 1), min((x + PS // 2) // 32, det.shape[1] - 1)]
            cand[int(c)].append((y, x, float(d)))
    picks = []
    want = {k: int(round(n * mix[k])) for k in (0, 1, 2)}
    # spare quota from missing categories goes to detail first, then flat
    for k in (0, 1):
        short = max(0, want[k] - len(cand[k])); want[k] -= short; want[2] += short
    short = max(0, want[2] - len(cand[2])); want[2] -= short; want[1] = min(len(cand[1]), want[1] + short)
    for k in (0, 1):
        if want[k]:
            for i in rng.choice(len(cand[k]), want[k], replace=False):
                picks.append((cand[k][i][0], cand[k][i][1], k))
    if want[2]:
        d = np.array([c[2] for c in cand[2]]) + 1e-3
        for i in rng.choice(len(cand[2]), want[2], replace=False, p=d / d.sum()):
            picks.append((cand[2][i][0], cand[2][i][1], 2))
    return picks


def build_bank(a, files, per_page, mix, path, seed):
    rng = np.random.default_rng(seed)
    n_pages = sum(len(read_names(f)) for f in files)
    if a.pages_per_cbz:
        n_pages = min(n_pages, a.pages_per_cbz * len(files))
    cap = n_pages * per_page
    LR = np.lib.format.open_memmap(path + "_lr.npy", "w+", np.uint8, (cap, PS, PS, 3))
    HR = np.lib.format.open_memmap(path + "_hr.npy", "w+", np.uint8, (cap, 2 * PS, 2 * PS, 3))
    cats = []
    for f in files:
        pages = read_cbz(f)[: a.pages_per_cbz] if a.pages_per_cbz else read_cbz(f)
        for name, lr in tqdm(pages, desc=f"bank {os.path.basename(path)} <- {os.path.basename(f)}", unit="page"):
            hr = teacher_img(a, f, name)
            assert hr.shape[:2] == (lr.shape[0] * 2, lr.shape[1] * 2), f"teacher size mismatch on {f}/{name}"
            for y, x, c in sample_positions(lr, per_page, rng, mix):
                k = len(cats)
                LR[k] = lr[y:y + PS, x:x + PS]; HR[k] = hr[2 * y:2 * y + 2 * PS, 2 * x:2 * x + 2 * PS]; cats.append(c)
            del hr
    LR.flush(); HR.flush(); del LR, HR
    np.save(path + "_cat.npy", np.array(cats, np.uint8))  # its length = number of valid rows in _lr/_hr
    counts = np.bincount(np.array(cats), minlength=3)
    print(f"{os.path.basename(path)}: {len(cats)} patches | " + ", ".join(f"{CATS[k]} {counts[k]}" for k in range(3)), flush=True)


def stage_bank(a):
    train, test = split_files(a.data)
    bdir = os.path.join(a.out, "bank"); os.makedirs(bdir, exist_ok=True)
    if not os.path.exists(os.path.join(bdir, "train_cat.npy")):
        build_bank(a, train, a.patches_per_page, {0: 0.05, 1: 0.15, 2: 0.80}, os.path.join(bdir, "train"), 1)
    if not os.path.exists(os.path.join(bdir, "val_cat.npy")):
        build_bank(a, [test], a.val_patches_per_page, {0: 0.2, 1: 0.4, 2: 0.4}, os.path.join(bdir, "val"), 2)


# ---------------------------------------------------------------- stage: train
def stage_train(a):
    import torch, torch.nn as nn
    torch.backends.cudnn.benchmark = DEV == "cuda"
    bdir = os.path.join(a.out, "bank"); cdir = os.path.join(a.out, "ckpt"); os.makedirs(cdir, exist_ok=True)
    nt = len(np.load(os.path.join(bdir, "train_cat.npy")))
    lr_b = np.load(os.path.join(bdir, "train_lr.npy"), mmap_mode="r")[:nt]; hr_b = np.load(os.path.join(bdir, "train_hr.npy"), mmap_mode="r")[:nt]
    vc = np.load(os.path.join(bdir, "val_cat.npy"))
    vl = np.array(np.load(os.path.join(bdir, "val_lr.npy"), mmap_mode="r")[:len(vc)])
    vh = np.array(np.load(os.path.join(bdir, "val_hr.npy"), mmap_mode="r")[:len(vc)])
    N = nt
    print(f"train patches {N}, val patches {len(vc)}", flush=True)

    from spandrel import ModelLoader
    desc = ModelLoader().load_from_file(a.init)
    assert desc.scale == 2, "student should be a 2x model"
    model = desc.model.to(DEV).float()
    for m in model.modules():  # in-place activations can break backward
        if hasattr(m, "inplace"):
            m.inplace = False
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0)
    warm = min(500, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))) * (1 - 0.01) + 0.01)
    rng = np.random.default_rng(3)
    to_dev = lambda arr: torch.from_numpy(np.ascontiguousarray(arr[..., ::-1].transpose(0, 3, 1, 2))).to(DEV).float().div_(255)

    def batch():
        idx = rng.integers(0, N, a.batch)
        oy, ox = rng.integers(0, 2 * J + 1, 2)
        lr = lr_b[np.sort(idx)][:, oy:oy + P, ox:ox + P]
        hr = hr_b[np.sort(idx)][:, 2 * oy:2 * oy + 2 * P, 2 * ox:2 * ox + 2 * P]
        k = int(rng.integers(0, 4)); flip = bool(rng.integers(0, 2))
        lr = np.rot90(lr, k, axes=(1, 2)); hr = np.rot90(hr, k, axes=(1, 2))
        if flip:
            lr = lr[:, :, ::-1]; hr = hr[:, :, ::-1]
        return to_dev(lr), to_dev(hr)

    def edge_l1(o, y):
        """L1 between brightness gradients: makes softened lines/hatching expensive (plain L1 averages them away)."""
        w = torch.tensor([0.299, 0.587, 0.114], device=o.device).view(1, 3, 1, 1)
        lo, ly = (o * w).sum(1), (y * w).sum(1)
        return ((lo[..., 1:] - lo[..., :-1]) - (ly[..., 1:] - ly[..., :-1])).abs().mean() + \
               ((lo[..., 1:, :] - lo[..., :-1, :]) - (ly[..., 1:, :] - ly[..., :-1, :])).abs().mean()

    def validate():
        model.eval(); tot = np.zeros(3); cnt = np.zeros(3); psnr = []
        with torch.inference_mode():
            for i in range(0, len(vc), 64):
                x = to_dev(vl[i:i + 64, J:J + P, J:J + P]); y = to_dev(vh[i:i + 64, 2 * J:2 * J + 2 * P, 2 * J:2 * J + 2 * P])
                with torch.autocast(DEV, dtype=torch.bfloat16, enabled=a.bf16 and DEV == "cuda"):
                    o = model(x)
                err = (o.float().clamp(0, 1) - y).abs().mean(dim=(1, 2, 3)).cpu().numpy()
                mse = ((o.float().clamp(0, 1) - y) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10).cpu().numpy()
                psnr += list(10 * np.log10(1 / mse))
                for c in range(3):
                    m = vc[i:i + 64] == c; tot[c] += err[m].sum(); cnt[c] += m.sum()
        model.train()
        per = {CATS[c]: round(float(tot[c] / max(cnt[c], 1)) * 255, 3) for c in range(3)}
        return {"val_l1_levels": round(float(tot.sum() / cnt.sum()) * 255, 3), "per_cat_l1_levels": per, "val_psnr": round(float(np.mean(psnr)), 2)}

    log_path = os.path.join(a.out, "train_log.jsonl")
    start = 0; best = float("inf")
    last = os.path.join(cdir, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location=DEV)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, best = st["step"], st["best"]; print(f"resuming at step {start}", flush=True)
    log = open(log_path, "a")
    if start == 0:
        v = validate(); v.update(step=0, note="starting JaNai SPAN weights"); log.write(json.dumps(v) + "\n"); log.flush()
        print(f"step 0 (JaNai start): val diff {v['val_l1_levels']} levels {v['per_cat_l1_levels']}", flush=True)
        best = v["val_l1_levels"]  # training has to beat the starting weights to replace best.pth
        model.eval(); torch.save(model.state_dict(), os.path.join(cdir, "best.pth")); model.train()
    model.train()
    bar = tqdm(range(start, a.steps), initial=start, total=a.steps, desc="train", unit="step")
    run = 0.0; t0 = time.perf_counter()
    for step in bar:
        x, y = batch()
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=a.bf16 and DEV == "cuda"):
            o = model(x)
        o = o.float().clamp(0, 1)  # outputs are clipped at use time, so don't punish "below black": fixes the +1 black offset
        pix = (o - y).abs().mean()  # logged as train_l1_levels (comparable across runs)
        loss = pix + a.edge_weight * edge_l1(o, y) if a.edge_weight else pix
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * pix.item() if step > start else pix.item()
        if (step + 1) % 50 == 0:
            bar.set_postfix(loss_levels=f"{run * 255:.2f}", lr=f"{sched.get_last_lr()[0]:.1e}")
        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            v = validate(); v.update(step=step + 1, train_l1_levels=round(run * 255, 3),
                                     steps_per_s=round((step + 1 - start) / (time.perf_counter() - t0), 2))
            log.write(json.dumps(v) + "\n"); log.flush()
            tqdm.write(f"step {step + 1}: train {run * 255:.2f} | val {v['val_l1_levels']} levels {v['per_cat_l1_levels']} | psnr {v['val_psnr']}")
            model.eval(); sd = model.state_dict(); model.train()  # eval() refreshes the fused weights before saving
            if v["val_l1_levels"] < best:
                best = v["val_l1_levels"]; torch.save(sd, os.path.join(cdir, "best.pth"))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1, "best": best}, last)
    log.close()
    model.eval(); torch.save(model.state_dict(), os.path.join(cdir, "final.pth"))
    print(f"train: done, best val diff {best} levels", flush=True)


# ---------------------------------------------------------------- stage: eval
def stage_eval(a):
    import torch
    import exp06_split_vs_full as e6
    import exp15_native2x as e15
    torch.backends.cudnn.benchmark = DEV == "cuda"
    _, test = split_files(a.data, a.stage != "exam")
    edir = os.path.join(a.out, "eval"); os.makedirs(edir, exist_ok=True)
    # final.pth, not best.pth: best is chosen using patches from these same test pages, so evaluating it here
    # would flatter the score (selection leak). The cosine schedule ends near the best point anyway.
    final = os.path.join(a.out, "ckpt", "final.pth")
    student = load_model(a.student or (final if os.path.exists(final) else os.path.join(a.out, "ckpt", "best.pth")))
    start = load_model(a.init)
    times = json.load(open(os.path.join(a.out, "teacher", "times.json")))
    pages = read_cbz(test)[: a.eval_pages] if a.eval_pages else read_cbz(test)
    labels = {"today": "Today / teacher (4x-UltraSharp, halved)", "student": "Student (trained SPAN)", "start": "JaNai SPAN (before training)"}
    rows = []
    for i, (name, img) in enumerate(tqdm(pages, desc="eval", unit="page")):
        tea = teacher_img(a, test, name)
        outs = {"today": tea}; rec = {"page": name, "t_today": times.get(f"test.cbz/{name}")}
        for key, m in (("student", student), ("start", start)):
            run_model(m, img[:256]); sync(); t = time.perf_counter()
            o = run_model(m, img); sync(); rec[f"t_{key}"] = round(time.perf_counter() - t, 3)
            outs[key] = o
            smap = e6.ssim_map(tea, o)
            rec[f"ssim_{key}"] = round(float(smap.mean()), 4)
            d = cv2.absdiff(tea, o); rec[f"meandiff_{key}"] = round(float(d.mean()), 2); rec[f"maxdiff_{key}"] = int(d.max())
            cat, _ = tile_categories(img, 32)
            big = np.kron(cat, np.ones((64, 64), np.uint8))[:smap.shape[0], :smap.shape[1]]
            for c in range(3):
                m_ = big == c
                if m_.any():
                    rec[f"ssim_{key}_{CATS[c]}"] = round(float(smap[:big.shape[0], :big.shape[1]][m_].mean()), 4)
        if rec["t_today"] is None:
            rec["t_today"] = 0
        e15.write_compare(os.path.join(edir, f"compare_page{i:02d}.html"), i, f"Exp 16 test page {name}", img, outs, labels, rec, full=True)  # whole page, always
        rows.append(rec)
    json.dump(rows, open(os.path.join(edir, "eval.json"), "w"), indent=1)
    mean = lambda k: np.mean([r[k] for r in rows if k in r])
    lines = [f"{len(rows)} held-out test pages. Similarity/difference are vs the teacher (today's look): here that IS the quality score."]
    for key in ("student", "start"):
        lines.append(f"{labels[key]:30s}: similarity {mean(f'ssim_{key}'):.4f} (blank {mean(f'ssim_{key}_blank'):.4f}, "
                     f"flat {mean(f'ssim_{key}_flat'):.4f}, detail {mean(f'ssim_{key}_detail'):.4f}) | mean diff "
                     f"{mean(f'meandiff_{key}'):.2f} levels, worst max {max(r[f'maxdiff_{key}'] for r in rows)} | {mean(f't_{key}'):.2f}s per page")
    if any(r["t_today"] for r in rows):
        lines.append(f"Teacher on this GPU: {mean('t_today'):.2f}s per page (model only)")
    text = "\n".join(lines); print("\n" + text, flush=True)
    open(os.path.join(a.out, "summary.txt"), "w").write(text + "\n")
    lp, bp = os.path.join(a.out, "train_log.jsonl"), os.path.join(a.out, "bank", "train_cat.npy")
    log = [json.loads(l) for l in open(lp) if l.strip()] if os.path.exists(lp) else []
    best = min((r for r in log if r["step"] > 0), key=lambda r: r["val_l1_levels"], default=None)
    res = {"run": os.path.basename(os.path.normpath(a.out)), "args": vars(a), "student_ckpt": a.student or final,
           "train_patches": int(len(np.load(bp))) if os.path.exists(bp) else None,
           "start_val": log[0] if log else None, "best_val": best, "test_pages": len(rows)}
    for key in ("student", "start"):
        res[key] = {m: round(float(mean(f"{m}_{key}")), 4) for m in ("ssim", "meandiff", "t")}
        res[key].update({f"ssim_{c}": round(float(mean(f"ssim_{key}_{c}")), 4) for c in CATS})
    res["teacher_t"] = round(float(mean("t_today")), 3) if any(r["t_today"] for r in rows) else None
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["teacher", "bank", "train", "eval", "all", "exam"],
                    help="exam = teacher + eval only, on <data>/test.cbz (no training files needed); use with --student")
    ap.add_argument("--data", default="data/training/v3")
    ap.add_argument("--out", default="outputs/exp16-distill")
    ap.add_argument("--teacher", default="models/4x-UltraSharp.pth")
    ap.add_argument("--init", default="models/2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.safetensors")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--edge-weight", type=float, default=0.5, help="weight of the edge (gradient) loss; 0 = plain L1 as in run 1")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--patches-per-page", type=int, default=120)
    ap.add_argument("--val-patches-per-page", type=int, default=40)
    ap.add_argument("--pages-per-cbz", type=int, default=0, help="limit pages (0 = all)")
    ap.add_argument("--eval-pages", type=int, default=0)
    ap.add_argument("--student", default="", help="checkpoint to evaluate (default: <out>/ckpt/final.pth)")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 3 pages per CBZ, 300 steps, batch 8")
    a = ap.parse_args()
    if a.smoke:
        a.pages_per_cbz, a.steps, a.batch, a.val_every, a.eval_pages = 3, 300, 8, 100, 2
        a.patches_per_page, a.val_patches_per_page = 60, 30
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    stages = {"all": ["teacher", "bank", "train", "eval"], "exam": ["teacher", "eval"]}.get(a.stage, [a.stage])
    for s in stages:
        t = time.perf_counter()
        {"teacher": stage_teacher, "bank": stage_bank, "train": stage_train, "eval": stage_eval}[s](a)
        print(f"== {s} took {(time.perf_counter() - t) / 60:.1f} min", flush=True)
    open(os.path.join(a.out, "DONE" if a.stage == "all" else f"DONE_{a.stage}"), "w").write("ok\n")


if __name__ == "__main__":
    main()
