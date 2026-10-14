"""Exp 16: run the unseen-series exam on several series in one go, then write one summary table.

For each series: build exams/<slug>/test.cbz (one random page per chapter, 30 chapters), run the teacher and
the eval (student vs JaNai, whole-page comparers). Series already examined are skipped (DONE_exam), so it is
safe to re-run after an interruption. Results:
  experiments/out/exp16_exam/<slug>_<tag>/            per-series comparers, eval.json, summary.txt
  experiments/out/exp16_exam/exams_<tag>.csv          one row per series (for the README chart)

Series names never appear in the repository. They live in two gitignored files on this machine:
  exams/series.txt          the series folders to examine, one per line (folder names under Manwhas/)
  exams/series_key.json     {"A": {"style": "Soft pastel, painted", "series": "<series folder>"}, ...}
The CSV and everything printed use the anonymous label and style from the key instead of the name.

Run from the repo root (takes ~10 min per series on the laptop, mostly the teacher):
  python experiments/exp16/run_exams.py
"""
import argparse, csv, json, os, re, subprocess, sys

SERIES_LIST = "exams/series.txt"
SERIES_KEY = "exams/series_key.json"
RUN2 = "experiments/out/exp16_cloud/run_20261013_0044/ckpt/final.pth"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def load_series():
    """The series to examine (from SERIES_LIST) and, for each, its anonymous (label, style) from SERIES_KEY."""
    if not os.path.exists(SERIES_LIST) or not os.path.exists(SERIES_KEY):
        sys.exit(f"needs {SERIES_LIST} (one series folder per line) and {SERIES_KEY} (label -> style and series); both stay local")
    series = [line.strip() for line in open(SERIES_LIST) if line.strip()]
    key = json.load(open(SERIES_KEY))
    labels = {entry["series"]: (label, entry["style"]) for label, entry in key.items()}
    missing = [s for s in series if s not in labels]
    if missing:
        sys.exit(f"{len(missing)} series in {SERIES_LIST} have no label in {SERIES_KEY}")
    return series, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=RUN2)
    ap.add_argument("--tag", default="run2", help="name for this student in output folders")
    ap.add_argument("--series", default="", help="comma-separated labels from the key (e.g. A,C); default: every series in the list")
    a = ap.parse_args()
    series, labels = load_series()
    if a.series:
        wanted = {x.strip() for x in a.series.split(",") if x.strip()}
        series = [s for s in series if labels[s][0] in wanted]
    rows = []
    for i, s in enumerate(series, 1):
        sl = slug(s)
        label, style = labels[s]
        out = f"experiments/out/exp16_exam/{sl}_{a.tag}"
        print(f"\n===== [{i}/{len(series)}] {label} ({style})", flush=True)
        if not os.path.exists(os.path.join(out, "DONE_exam")):
            if not os.path.exists(f"exams/{sl}/test.cbz"):
                subprocess.run([sys.executable, "experiments/exp16/make_dataset.py", "--exam", s], check=True)
            subprocess.run([sys.executable, "experiments/exp16/distill.py", "exam", "--data", f"exams/{sl}",
                            "--out", out, "--student", a.student], check=True)
        else:
            print("already done, skipping", flush=True)
        ev = json.load(open(os.path.join(out, "eval", "eval.json")))
        m = lambda k: round(sum(r[k] for r in ev if k in r) / max(1, sum(1 for r in ev if k in r)), 4)
        rows.append({"series": label, "style": style, "pages": len(ev),
                     "student_ssim": m("ssim_student"), "janai_ssim": m("ssim_start"),
                     "student_detail": m("ssim_student_detail"), "janai_detail": m("ssim_start_detail"),
                     "student_flat": m("ssim_student_flat"), "janai_flat": m("ssim_start_flat"),
                     "student_blank": m("ssim_student_blank"), "janai_blank": m("ssim_start_blank"),
                     "detail_better_pages": sum(r["ssim_student_detail"] > r["ssim_start_detail"] for r in ev if "ssim_student_detail" in r),
                     "student_meandiff": m("meandiff_student"), "janai_meandiff": m("meandiff_start"),
                     "teacher_s_per_page": m("t_today"), "student_s_per_page": m("t_student"), "janai_s_per_page": m("t_start")})
    path = f"experiments/out/exp16_exam/exams_{a.tag}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'series':36s} detail student/JaNai  better  | overall student/JaNai | s/page teacher/student")
    for r in rows:
        print(f"{r['series'] + '  ' + r['style']:36s} {r['student_detail']:.4f} / {r['janai_detail']:.4f}  {r['detail_better_pages']:2d}/{r['pages']}"
              f"  | {r['student_ssim']:.4f} / {r['janai_ssim']:.4f} | {r['teacher_s_per_page']:.1f} / {r['student_s_per_page']:.2f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
