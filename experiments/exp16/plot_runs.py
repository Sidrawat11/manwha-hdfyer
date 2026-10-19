"""Exp 16: one ledger + README charts for every distillation run.

Reads every run folder in outputs/exp16-cloud/ (train_log.jsonl, eval/eval.json, args.json,
cloud.json, results.json when present) and writes:
  experiments/exp16/runs.csv            one row per run: setup, cost, best step, test scores, speed
  assets/exp16/val_curve.png            held-out error vs training step (overall and detail), per run
  assets/exp16/detail_per_page.png      per test page: detail similarity before vs after training
  assets/exp16/speed.png                seconds per page on the training GPU: teacher, JaNai, each student
Both locations are committed, so the README can embed the charts.

Run from the repo root after a run downloads:  python experiments/exp16/plot_runs.py
"""
import csv, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "outputs/exp16-cloud"
ASSETS = "assets/exp16"
LEDGER = "experiments/exp16/runs.csv"
# validated categorical order (blue, orange, aqua, yellow, ...) on a light surface; text never in series colour
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
CATS = ("blank", "flat", "detail")


def load_run(d):
    log = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl")) if l.strip()]
    ev = json.load(open(os.path.join(d, "eval", "eval.json"))) if os.path.exists(os.path.join(d, "eval", "eval.json")) else []
    rd = lambda f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else {}
    args, cloud = rd("args.json"), rd("cloud.json")
    m = lambda k: sum(r[k] for r in ev if k in r) / max(1, sum(1 for r in ev if k in r))
    best = min((r for r in log if r["step"] > 0), key=lambda r: r["val_l1_levels"], default=None)
    edge = args.get("edge_weight", 0)
    data = args.get("data", "").rstrip("/")
    manifest = os.path.join(data, "manifest.json")
    pages = sum(len(v) for v in json.load(open(manifest))["train"].values()) if os.path.exists(manifest) else "?"
    return {"name": os.path.basename(d), "log": log, "eval": ev, "args": args, "cloud": cloud, "best": best,
            "label": f"{os.path.basename(d)[4:17]}: {pages} pages, {'edge loss' if edge else 'plain L1'}",
            "row": {
                "run": os.path.basename(d), "train_pages": pages, "steps": args.get("steps"),
                "edge_weight": edge, "gpu": cloud.get("gpu", ""), "usd_per_h": cloud.get("dph", ""),
                "cost_usd_approx": cloud.get("cost_usd_approx", ""),
                "best_step": best and best["step"], "best_val_l1": best and best["val_l1_levels"],
                "best_val_detail_l1": best and best["per_cat_l1_levels"]["detail"],
                "start_val_detail_l1": log[0]["per_cat_l1_levels"]["detail"] if log else "",
                "student_ssim": round(m("ssim_student"), 4), "student_ssim_detail": round(m("ssim_student_detail"), 4),
                "student_ssim_blank": round(m("ssim_student_blank"), 4), "student_meandiff": round(m("meandiff_student"), 2),
                "janai_ssim_detail": round(m("ssim_start_detail"), 4), "janai_meandiff": round(m("meandiff_start"), 2),
                "student_s_per_page": round(m("t_student"), 3), "janai_s_per_page": round(m("t_start"), 3),
                "teacher_s_per_page": round(m("t_today"), 3),
            }}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    runs = [load_run(d) for d in sorted(glob.glob(os.path.join(RUNS, "run_*"))) if os.path.exists(os.path.join(d, "train_log.jsonl"))]
    if not runs:
        raise SystemExit(f"no runs in {RUNS}")
    os.makedirs(ASSETS, exist_ok=True)
    with open(LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]["row"].keys())); w.writeheader()
        for r in runs:
            w.writerow(r["row"])

    # 1. held-out error vs step: overall | detail (same y-scale), JaNai start as a grey reference
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, key, title in ((axes[0], None, "All test patches"), (axes[1], "detail", "Detailed patches (hatching, hair, texture)")):
        get = (lambda r: r["val_l1_levels"]) if key is None else (lambda r: r["per_cat_l1_levels"][key])
        start = get(runs[0]["log"][0])
        ax.axhline(start, color=REF, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.annotate("JaNai before training", (0, start), xytext=(4, 4), textcoords="offset points", color=INK2, fontsize=9)
        for i, r in enumerate(runs):
            xs = [p["step"] for p in r["log"]]; ys = [get(p) for p in r["log"]]
            ax.plot(xs, ys, color=SERIES[i], linewidth=2, label=r["label"])
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")
        style(ax, title, "training step", "difference from Today (grey levels, lower is better)" if key is None else "")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "val_curve.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

    # 2. per test page: detail similarity before (x) vs after (y); above the diagonal = improved
    fig, ax = plt.subplots(figsize=(5.6, 5.4), facecolor=SURFACE)
    allv = [v for r in runs for p in r["eval"] for v in (p.get("ssim_start_detail"), p.get("ssim_student_detail")) if v]
    lo, hi = min(allv) - 0.005, 1.0
    ax.plot([lo, hi], [lo, hi], color=REF, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate("no change", (hi, hi), xytext=(-60, -14), textcoords="offset points", color=INK2, fontsize=9)
    for i, r in enumerate(runs):
        pts = [(p["ssim_start_detail"], p["ssim_student_detail"]) for p in r["eval"] if "ssim_student_detail" in p]
        if pts:
            better = sum(1 for a, b in pts if b > a)
            ax.scatter(*zip(*pts), s=36, color=SERIES[i], edgecolor=SURFACE, linewidth=1.2,
                       label=f"{r['label']} ({better}/{len(pts)} pages better)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    style(ax, "Detail similarity to Today, per test page", "JaNai before training", "student after training")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "detail_per_page.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

    # 3. speed on the training GPU (model only, one strip at a time): single series, one colour
    names, vals = ["Today (4x-UltraSharp)", "JaNai SPAN"], [runs[-1]["row"]["teacher_s_per_page"], runs[-1]["row"]["janai_s_per_page"]]
    for r in runs:
        names.append(f"Student {r['name'][4:17]}"); vals.append(r["row"]["student_s_per_page"])
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.2), facecolor=SURFACE)
    ax.barh(names[::-1], vals[::-1], color=SERIES[0], height=0.55)
    for y, v in enumerate(vals[::-1]):
        ax.annotate(f"{v:.2f} s", (v, y), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")
    gpu = runs[-1]["cloud"].get("gpu") or "training GPU"
    style(ax, f"Seconds per page on {gpu} (model only, unbatched)", "seconds per page", "")
    ax.grid(axis="y", visible=False); ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "speed.png"), dpi=150, facecolor=SURFACE); plt.close(fig)
    print(f"{len(runs)} run(s) -> {LEDGER}, {ASSETS}/val_curve.png, detail_per_page.png, speed.png")


if __name__ == "__main__":
    main()
