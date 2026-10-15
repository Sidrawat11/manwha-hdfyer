"""Exp 17: how fast is the shipped pipeline on a given GPU? Runs anywhere: the laptop or a rented H200.

Drives hdfyer.pipeline.process_all exactly as production does (hdfyer, TensorRT, CPU overlap), each config in a fresh
process, on a fixed 3-chapter set covering 3 page widths. Every config runs twice: the first run pays for
engine builds and the batch profile (reported separately), the second is the timed one.

  batch sweep    TensorRT at batch 1, 2, 4, 8, 16 with plenty of CPU workers, so the GPU is the only limit
  auto           --batch auto: what the profiler picks on this GPU
  worker sweep   at the best batch: 1, 2, 4, 8, 16 CPU workers (a fast GPU can outrun the CPU finishing)
  eager          batch 1, no TensorRT, for reference
  parallel       with --parallel "1,2,4,...": only this sweep. N chapters at once in N worker processes
                 (main.py --parallel N), batch 1, 4 CPU workers per process; total pages/s and s/page per N
nvidia-smi dmon runs alongside and is saved, so GPU busy/idle can be read per second.

Outputs (no page images): <out>/results.json, <out>/summary.txt, <out>/logs/*.log, <out>/dmon.log

Run from the repo root:
  python experiments/exp17/bench.py --chapters bench_chapters --out experiments/out/exp17/laptop
  python experiments/exp17/bench.py --chapters bench_chapters_par --parallel 1,2,4,6,8 --out experiments/out/exp17/par
"""
import argparse, json, os, re, shutil, subprocess, sys, time

RUN = """
import sys, time; sys.path.insert(0, '.')
from pathlib import Path
try:
    from hdfyer.pipeline import process_all   # the hdfyer/ package layout
except ImportError:
    from main import process_all              # --code-dir on a commit from before it (tag pre-restructure)
t = time.perf_counter()
process_all(Path({inp!r}), Path({outp!r}), backend={backend!r}, batch={batch!r},
            cpu_workers={workers}, max_inflight={inflight}, overlap=True, parallel={parallel})
"""


CODE_DIR = "."  # --code-dir: the checkout whose pipeline is benchmarked (paths below are made absolute)


def run(name, chapters, out, backend, batch, workers, parallel=1):
    """One config in a fresh process; returns parsed timings."""
    chapters = os.path.abspath(chapters)
    res_dir = os.path.abspath(os.path.join(out, "tmp_results", name))
    shutil.rmtree(res_dir, ignore_errors=True)
    code = RUN.format(inp=chapters, outp=res_dir, backend=backend, batch=batch, workers=workers, inflight=workers + 2,
                      parallel=parallel)
    t = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=CODE_DIR,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    wall = time.perf_counter() - t
    log = (p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or "")
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    open(os.path.join(out, "logs", f"{name}.log"), "w").write(log)
    shutil.rmtree(res_dir, ignore_errors=True)  # the upscaled chapters are not kept
    r = parse(log, parallel)
    r.update({"name": name, "backend": backend, "batch": batch, "workers": workers, "parallel": parallel,
              "process_wall_s": round(wall, 2), "ok": p.returncode == 0})
    if r.get("chapters_s") is not None:
        r["startup_s"] = round(wall - r["chapters_s"], 2)
    return r


def parse(log, parallel=1):
    """Pull the numbers main.py prints."""
    r = {"per_chapter_s": {}}
    for m in re.finditer(r"Done: (.+?) in ([\d.]+)s", log):
        r["per_chapter_s"][m.group(1)] = float(m.group(2))
    m = re.search(r"Finished (\d+) chapters in ([\d.]+)s", log)
    if m:
        r["chapters"] = int(m.group(1))
        if parallel > 1:
            # Chapters overlap in time, so their times can't be summed: use the wall time of the worker phase.
            r["chapters_s"] = float(m.group(2))
        else:  # one at a time: the per-chapter lines (0.1 s resolution) add up to the chapter time
            r["chapters_s"] = round(sum(r["per_chapter_s"].values()), 2) if r["per_chapter_s"] else float(m.group(2))
    m = re.search(r"TensorRT engines built this run: (\d+), ([\d.]+)s", log)
    r["engines_built"], r["engine_build_s"] = (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)
    r["fallbacks"] = 0 if "Fallbacks: none" in log else len(re.findall(r"!!! FALLBACK", log))
    return r


def pages_in(chapters):
    import zipfile
    n = 0
    for root, _, files in os.walk(chapters):
        for f in files:
            if f.endswith(".cbz"):
                with zipfile.ZipFile(os.path.join(root, f)) as z:
                    n += sum(1 for x in z.namelist() if x.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters", default="bench_chapters")
    ap.add_argument("--out", default="experiments/out/exp17/laptop")
    ap.add_argument("--batches", default="1,2,4,8,16")
    ap.add_argument("--workers", default="1,2,4,8,16")
    ap.add_argument("--quick", action="store_true", help="batch 1 and auto only (smoke test)")
    ap.add_argument("--code-dir", default=".", help="checkout to benchmark (another commit's worktree); default: this one")
    ap.add_argument("--parallel", default="", help='e.g. "1,2,4,6,8": sweep main.py --parallel N only (batch 1, 4 CPU '
                                                   'workers per process); the batch and worker sweeps are skipped')
    a = ap.parse_args()
    global CODE_DIR
    CODE_DIR = os.path.abspath(a.code_dir)
    os.makedirs(a.out, exist_ok=True)
    ncpu = os.cpu_count() or 4
    many = min(16, ncpu)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    pages = pages_in(a.chapters)
    meta = {"gpu": gpu, "cpus": ncpu, "pages": pages, "chapters_dir": a.chapters, "started": time.strftime("%Y-%m-%d %H:%M")}
    print(f"GPU {gpu} | {ncpu} CPUs | {pages} pages", flush=True)
    dmon = subprocess.Popen(["nvidia-smi", "dmon", "-s", "pucm"], stdout=open(os.path.join(a.out, "dmon.log"), "w"))
    results = []

    def go(name, backend, batch, workers, parallel=1):
        for phase in ("warm", "timed"):  # warm pays engine builds / profiling; timed is the number we use
            r = run(f"{name}_{phase}", a.chapters, a.out, backend, batch, workers, parallel)
            r["phase"], r["config"] = phase, name
            results.append(r)
            if r.get("chapters_s"):
                r["s_per_page"] = round(r["chapters_s"] / pages, 4)
                r["pages_per_s"] = round(pages / r["chapters_s"], 2)
            print(f"  {name:14s} {phase:5s} chapters {r.get('chapters_s')} s | {r.get('s_per_page')} s/page | "
                  f"engines built {r['engines_built']} ({r['engine_build_s']:.0f}s) | startup {r.get('startup_s')} s | "
                  f"fallbacks {r['fallbacks']} | ok {r['ok']}", flush=True)
            json.dump({"meta": meta, "runs": results}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
            if not r["ok"]:
                print(open(os.path.join(a.out, "logs", f"{name}_{phase}.log")).read()[-3000:], flush=True)
                break

    try:
        if a.parallel:
            for n in [int(x) for x in a.parallel.split(",") if x.strip()]:
                go(f"parallel_{n}", "trt", 1, 4, n)
            return
        batches = [1] if a.quick else [int(x) for x in a.batches.split(",")]
        for b in batches:
            go(f"trt_b{b}_w{many}", "trt", b, many)
        go(f"trt_auto_w{many}", "trt", "auto", many)
        timed = [r for r in results if r["phase"] == "timed" and r.get("chapters_s") and r["backend"] == "trt"
                 and isinstance(r["batch"], int)]
        best = min(timed, key=lambda r: r["chapters_s"])["batch"] if timed else 1
        if not a.quick:
            for w in [int(x) for x in a.workers.split(",") if int(x) <= ncpu]:
                if w != many:
                    go(f"trt_b{best}_w{w}", "trt", best, w)
            go(f"eager_b1_w{many}", "eager", 1, many)
    finally:
        dmon.terminate()
        meta["finished"] = time.strftime("%Y-%m-%d %H:%M")
        json.dump({"meta": meta, "runs": results}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
        write_summary(a.out, meta, results)
        open(os.path.join(a.out, "DONE"), "w").write("ok\n")


def write_summary(out, meta, results):
    rows = [r for r in results if r["phase"] == "timed"]
    lines = [f"Exp 17 benchmark | {meta['gpu']} | {meta['cpus']} CPUs | {meta['pages']} pages in {meta['chapters_dir']}",
             "Timed run of each config (engines already built). Chapter time excludes process start-up "
             "(and, with --parallel, the warm-up).", "",
             f"{'config':16s} {'chapters s':>10s} {'s/page':>8s} {'pages/s':>8s} {'startup s':>9s} {'fallbacks':>9s}"]
    for r in rows:
        lines.append(f"{r['config']:16s} {r.get('chapters_s', float('nan')):10.1f} {r.get('s_per_page', float('nan')):8.3f} "
                     f"{r.get('pages_per_s', float('nan')):8.2f} {r.get('startup_s', float('nan')):9.1f} {r['fallbacks']:9d}")
    warm = [r for r in results if r["phase"] == "warm" and r["engines_built"]]
    if warm:
        lines += ["", "One-time engine builds (first sight of each width x batch):"]
        lines += [f"  {r['config']:16s} {r['engines_built']} engines, {r['engine_build_s']:.0f}s" for r in warm]
    if rows and rows[0].get("chapters_s"):
        pages = meta["pages"]
        best = min((r for r in rows if r.get("chapters_s")), key=lambda r: r["chapters_s"])
        lines += ["", f"Best: {best['config']} at {best['s_per_page']:.3f} s/page -> 1,000 chapters of ~45 pages "
                      f"in ~{best['s_per_page'] * 45000 / 60:.0f} min (excluding one-time engine builds)"]
    text = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(text + "\n")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
