"""README charts that are not tied to one experiment run. Same palette and style as experiments/exp16/plot_runs.py.

  assets/journey.png      seconds for one 34-page chapter at each step of the speed-up (RTX 4060 Laptop)
  assets/exp16/exam.png   detail similarity per unseen series: HDFyer vs the off-the-shelf JaNai SPAN
                          (from experiments/exp16/exams_run2.csv)

Run from the repo root:  python experiments/plot_readme.py
"""
import csv, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp16"))
from plot_runs import SERIES, SURFACE, INK, INK2, GRID, REF, style

# Wall time for one 34-page chapter, measured once per step on the same laptop, full pipeline incl. JPEG and CBZ.
JOURNEY = [
    ("Start", 680.1),
    ("Smarter strips", 589.2),
    ("TensorRT", 276.2),
    ("CPU work overlapped\nwith the GPU", 253.3),
    ("HDFyer + TensorRT", 10.9),
]


def journey():
    names, secs = [n for n, _ in JOURNEY], [s for _, s in JOURNEY]
    fig, ax = plt.subplots(figsize=(8, 3.9), facecolor=SURFACE)
    colors = [SERIES[0]] * (len(secs) - 1) + [SERIES[1]]
    ax.barh(names[::-1], secs[::-1], color=colors[::-1], height=0.58)
    for y, s in enumerate(secs[::-1]):
        label = f"{s:.0f} s" if s >= 100 else f"{s:.1f} s"
        speedup = secs[0] / s
        if speedup > 1.05:
            label += f"  ({speedup:.1f}x)" if speedup < 10 else f"  ({speedup:.0f}x)"
        ax.annotate(label, (s, y), xytext=(5, 0), textcoords="offset points", color=INK2, fontsize=9.5, va="center")
    style(ax, "One 34-page chapter on an RTX 4060 Laptop", "seconds for the whole chapter (lower is better)", "")
    ax.tick_params(axis="y", labelcolor=INK)
    ax.grid(axis="y", visible=False); ax.set_xlim(0, max(secs) * 1.22)
    fig.tight_layout(); fig.savefig("assets/journey.png", dpi=150, facecolor=SURFACE); plt.close(fig)


def exam():
    rows = list(csv.DictReader(open("experiments/exp16/exams_run2.csv")))
    rows.sort(key=lambda r: float(r["student_detail"]))
    fig, ax = plt.subplots(figsize=(8, 3.9), facecolor=SURFACE)
    for y, r in enumerate(rows):
        j, s = float(r["janai_detail"]), float(r["student_detail"])
        ax.plot([j, s], [y, y], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        ax.scatter([j], [y], s=70, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2, zorder=2, label="JaNai SPAN (off the shelf)" if y == 0 else None)
        ax.scatter([s], [y], s=70, color=SERIES[1], edgecolor=SURFACE, linewidth=1.2, zorder=3, label="HDFyer (ours)" if y == 0 else None)
        ax.annotate(f"better on {r['detail_better_pages']}/{r['pages']} pages", (max(j, s), y), xytext=(10, 0),
                    textcoords="offset points", color=INK2, fontsize=9, va="center")
    # Series are anonymised: label and drawing style only (the key stays local, see experiments/exp16/run_exams.py).
    ax.set_yticks(range(len(rows)), [f"{r['series']}  {re.sub(r' *[(].*[)]', '', r['style'])}" for r in rows])
    lo = min(float(r["janai_detail"]) for r in rows)
    ax.set_xlim(lo - 0.01, 1.04)
    style(ax, "Five series HDFyer never saw in training, 30 pages each",
          "similarity to the target look in detailed areas (SSIM, higher is better)", "")
    ax.tick_params(axis="y", labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout(); fig.savefig("assets/exp16/exam.png", dpi=150, facecolor=SURFACE); plt.close(fig)


if __name__ == "__main__":
    journey(); exam()
    print("wrote assets/journey.png, assets/exp16/exam.png")
