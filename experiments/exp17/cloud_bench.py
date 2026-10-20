"""Exp 17 on a rented Vast.ai GPU: rent -> upload code + weights + benchmark chapters -> bench.py -> download -> destroy.

Reuses Exp 16's Vast plumbing (offer search, SSH, always-destroy). Uploads the code as committed
(git archive HEAD), the hdfyer weights, and the chapters in data/bench/three-widths/ (~120 MB; they live only on the rented
machine and are deleted with it). TensorRT engines are NOT uploaded: they are tied to the GPU, so the server
builds its own and the warm-up runs report how long that takes.

Downloads only numbers and logs: results.json, summary.txt, logs/, dmon.log, plus the server's
cache/<gpu>.json (its measured batch profile) so the repo gains that GPU's profile.

Run from the repo root (WSL venv with vastai set up, as for Exp 16):
  python experiments/exp17/cloud_bench.py --dry-run
  python experiments/exp17/cloud_bench.py                       # H200 first, then H100s
  python experiments/exp17/cloud_bench.py --quick               # smoke test: batch 1 + auto only
  python experiments/exp17/cloud_bench.py --chapters-dir data/bench/sixteen --parallel 1,2,4,6,8
"""
import argparse, os, shlex, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cloud"))
import vast as cr  # shared Vast.ai helpers: sh, vast, write_cloud_info

IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"  # same torch as the laptop: nothing big to install, no 2.5.1 shadowing
REMOTE = "/workspace/m"
WEIGHTS = "models/hdfyer/hdfyer-2x-s-r2.pth"
BENCH_DIR = "data/bench/three-widths"  # default --chapters-dir; gitignored: the benchmark chapters as <series label>/<chapter>.cbz


def find_chapters(chapters_dir=BENCH_DIR):
    out = []
    for root, _, files in os.walk(chapters_dir):
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".cbz")]
    return sorted(out)


# Same versions as the laptop (requirements.txt). cu128 wheels need a driver that supports CUDA 12.8.
# Installed with --no-deps (vast.install_packages), so their own dependencies are listed too.
PACKAGES = cr.BASE_PACKAGES + ("imagesize", "tensorrt-cu12==11.4.0.106", "tensorrt_cu12_libs==11.4.0.106",
                               "tensorrt_cu12_bindings==11.4.0.106", "onnx==1.23.2", "protobuf", "ml_dtypes", "onnxscript", "onnx_ir")
ENGINES = "cache/trt_engines"  # engine files are named by GPU model + TensorRT version + model hash: reusable across machines


def engines_for(gpu_name):
    """Local engine files built on this GPU model (e.g. 'H200 NVL' -> ..._nvidia-h200-nvl_trt...)."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", (gpu_name or "").lower()).strip("-")
    pat = re.compile(rf"_nvidia-(?:geforce-)?{re.escape(slug)}_trt")
    return sorted(os.path.join(ENGINES, f) for f in os.listdir(ENGINES) if pat.search(f)) if slug and os.path.isdir(ENGINES) else []


def find_offer(gpus, max_dph, disk, geo=""):
    for g in gpus:
        q = (f"gpu_name={g} num_gpus=1 verified=true rentable=true direct_port_count>=1 reliability>0.98 "
             f"inet_down>=300 disk_space>={disk} dph_total<={max_dph} cuda_max_good>=12.8 cpu_cores_effective>=16"
             + (f" geolocation={geo}" if geo else ""))
        offers = cr.vast("search", "offers", q, "-o", "dph_total", check=False)
        if isinstance(offers, list) and offers:
            return offers[0]
        print(f"  no {g} under ${max_dph}/h right now", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="H200,H200_NVL,H100_SXM,H100_NVL,H100_PCIE")
    ap.add_argument("--max-dph", type=float, default=4.5)
    ap.add_argument("--geo", default="US")
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--disk", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--compare", default="", help="also benchmark this git ref (e.g. a9a42b6) on the SAME machine, in "
                                                  "its own checkout and engine cache; results in <run>/base/")
    ap.add_argument("--run", default=time.strftime("bench_%Y%m%d_%H%M"))
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chapters-dir", default=BENCH_DIR, help="local folder of benchmark chapters (uploaded as bench/)")
    ap.add_argument("--parallel", default="", help='passed to bench.py, e.g. "1,2,4,6,8": the --parallel sweep only')
    a = ap.parse_args()

    CHAPTERS = find_chapters(a.chapters_dir)
    if not CHAPTERS:
        sys.exit(f"no chapters in {a.chapters_dir}/ (put chapters of different page widths there)")
    if not os.path.exists(WEIGHTS):
        sys.exit(f"missing {WEIGHTS} (run from the repo root)")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "*.py"]).returncode != 0:
        print("NOTE: uncommitted .py changes are NOT uploaded (the server gets git HEAD).", flush=True)
    bal = cr.vast("show", "user").get("credit")
    print(f"Vast.ai credit: ${bal}", flush=True)
    offer = find_offer([g.strip() for g in a.gpus.split(",")], a.max_dph, a.disk, a.geo)
    if not offer:
        sys.exit("No offer matched. Raise --max-dph, drop --geo, or add GPUs.")
    dph = float(offer["dph_total"])
    print(f"Best offer: {offer.get('gpu_name')} {offer.get('gpu_ram', 0) / 1024:.0f} GB | "
          f"{offer.get('cpu_cores_effective', '?')} CPU cores | ${dph:.3f}/h | {offer.get('geolocation', '?')} | "
          f"offer {offer['id']}", flush=True)
    print(f"Expected ~30-45 min (~${dph * 0.6:.2f}); worst case at --max-hours {a.max_hours}: ${dph * a.max_hours:.2f}", flush=True)
    if a.dry_run:
        return
    if not a.yes and input("Rent it? [y/N] ").strip().lower() != "y":
        return

    res = cr.vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(a.disk), "--ssh", "--direct",
                  "--label", f"manwha-exp17-{a.run}")
    iid = res.get("new_contract") if isinstance(res, dict) else None
    if not iid:
        sys.exit(f"create failed: {res}")
    print(f"Rented instance {iid}. It will be destroyed when this script ends.", flush=True)
    t_start = time.time(); deadline = t_start + a.max_hours * 3600
    local_out = os.path.join("outputs", "exp17-bench", a.run)
    remote_py = {}  # the server's Python with torch, recorded in cloud.json
    try:
        while True:
            info = cr.vast("show", "instance", str(iid), check=False)
            st = info.get("actual_status") if isinstance(info, dict) else None
            print(f"  status: {st}", flush=True)
            if st == "running":
                break
            if time.time() - t_start > 900:
                raise RuntimeError("instance not running after 15 min")
            time.sleep(15)
        userhost, port = cr.sh(["vastai", "ssh-url", str(iid)]).split("://", 1)[1].rsplit(":", 1)
        ssh = ["ssh", "-n", "-p", port, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", userhost]
        for _ in range(40):
            if subprocess.run(ssh + ["true"], capture_output=True).returncode == 0:
                break
            time.sleep(10)
        else:
            raise RuntimeError("ssh never came up")
        py, torch_ver = cr.find_python(ssh)  # absolute path: a bare `python` over ssh may be the OS's, without torch
        remote_py.update(python=py, torch=torch_ver)
        print(f"  server Python: {py} (torch {torch_ver})", flush=True)

        print(f"  ssh ok; uploading code (git HEAD), weights and {len(CHAPTERS)} chapters...", flush=True)
        t = time.time()
        ssh_in = [x for x in ssh if x != "-n"]
        cr.sh(ssh + [f"mkdir -p {REMOTE}/bench"])
        git = subprocess.Popen(["git", "archive", "HEAD"], stdout=subprocess.PIPE)
        subprocess.run(ssh_in + [f"tar xf - -C {REMOTE}"], stdin=git.stdout, check=True); git.wait()
        tar = subprocess.Popen(["tar", "cf", "-", WEIGHTS, "experiments/exp17/bench.py"], stdout=subprocess.PIPE)  # bench.py: working copy, may be uncommitted
        subprocess.run(ssh_in + [f"tar xf - -C {REMOTE}"], stdin=tar.stdout, check=True); tar.wait()
        reuse = engines_for(offer.get("gpu_name"))
        if reuse:  # engines from an earlier run on this GPU model: skips the one-time builds (new code's cache only)
            tar_e = subprocess.Popen(["tar", "cf", "-", *reuse], stdout=subprocess.PIPE)
            subprocess.run(ssh_in + [f"mkdir -p {REMOTE}/{ENGINES} && tar xf - -C {REMOTE}"], stdin=tar_e.stdout, check=True); tar_e.wait()
            print(f"  reused {len(reuse)} TensorRT engines built on an earlier {offer.get('gpu_name')}", flush=True)
        if a.compare:  # the older commit in its own folder; shares only the weights (its engines may not be compatible)
            cr.sh(ssh + [f"mkdir -p {REMOTE}/base"])
            git_b = subprocess.Popen(["git", "archive", a.compare], stdout=subprocess.PIPE)
            subprocess.run(ssh_in + [f"tar xf - -C {REMOTE}/base"], stdin=git_b.stdout, check=True); git_b.wait()
            cr.sh(ssh + [f"mkdir -p {REMOTE}/base/{os.path.dirname(WEIGHTS)} && ln -sf {REMOTE}/{WEIGHTS} {REMOTE}/base/{WEIGHTS}"])  # link the FILE: models/hdfyer/ already exists there (model card)
        for c in CHAPTERS:  # flattened into bench/<series label>/<chapter>.cbz
            dst = f"{REMOTE}/bench/{shlex.quote(os.path.basename(os.path.dirname(c)))}"
            with open(c, "rb") as f:
                subprocess.run(ssh_in + [f"mkdir -p {dst} && cat > {dst}/{shlex.quote(os.path.basename(c))}"], stdin=f, check=True)
        print(f"  uploaded in {time.time() - t:.0f}s", flush=True)

        bench_args = (["--chapters", "bench", "--out", f"runs/{a.run}"] + (["--quick"] if a.quick else [])
                      + (["--parallel", a.parallel] if a.parallel else []))
        # `&` binds to the WHOLE && chain, so the chain is one background shell; its own stdout/stdin must not be the
        # ssh channel, or ssh waits for it to finish (the "stuck at installing" hang of 12-13 Oct).
        print("  installing TensorRT 11.4 + small packages on the server (torch 2.10 comes with the image)...", flush=True)
        remote_py.update(cr.install_packages(ssh, py, torch_ver, PACKAGES, log=f"{REMOTE}/pip.log"))  # also checks torch is unchanged
        print(f"  installed; torch {remote_py['torch']}, torchvision {remote_py['torchvision']}", flush=True)
        remote_cmd = (f"cd {REMOTE} && setsid nohup bash -c \""
                      f"{py} -c 'import torch, tensorrt, spandrel; print(torch.__version__, tensorrt.__version__, torch.cuda.get_device_name(0))' >> pip.log 2>&1 && "
                      + (f"PYTHONUNBUFFERED=1 {py} experiments/exp17/bench.py --code-dir base {' '.join(bench_args).replace(f'runs/{a.run}', f'runs/{a.run}/base')} > run.log 2>&1 && "
                         if a.compare else "") +  # old code first, so the new run's DONE marks the end of everything
                      f"PYTHONUNBUFFERED=1 {py} experiments/exp17/bench.py {' '.join(bench_args)} >> run.log 2>&1\" "
                      f"> /dev/null 2>&1 < /dev/null &")
        try:
            subprocess.run(ssh + [remote_cmd], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            pass
        for _ in range(18):  # up to 3 min until bench.py appears
            if subprocess.run(ssh + ["pgrep -f exp17/bench.py"], capture_output=True).returncode == 0:
                break
            time.sleep(10)
        else:
            raise RuntimeError("bench did not start:\n" + cr.sh(ssh + [f"tail -n 30 {REMOTE}/pip.log"], check=False))
        print("  " + cr.sh(ssh + [f"tail -n 1 {REMOTE}/pip.log"], check=False), flush=True)

        print("  benchmark running; progress every 30 s (Ctrl+C stops and destroys the instance)", flush=True)
        last = ""
        while True:
            time.sleep(30)
            out = subprocess.run(ssh + [f"cd {REMOTE}; grep -E '^  |^GPU|^Best' run.log | tail -n 2; "
                                        f"test -f runs/{a.run}/DONE && echo __DONE__; "
                                        f"pgrep -f exp17/bench.py >/dev/null || echo __DEAD__"],
                                 capture_output=True, text=True).stdout
            lines = [l for l in out.splitlines() if not l.startswith("__")]
            if lines and lines[-1] != last:
                last = lines[-1]
                print(f"  [{(time.time() - t_start) / 60:5.1f} min, ~${(time.time() - t_start) / 3600 * dph:.2f}] {last.strip()[:170]}", flush=True)
            if "__DONE__" in out:
                break
            if "__DEAD__" in out:
                print(cr.sh(ssh + [f"tail -n 40 {REMOTE}/run.log"], check=False), flush=True)
                break
            if time.time() > deadline - 300:
                print("  approaching --max-hours: downloading what exists, then destroying", flush=True)
                break

        os.makedirs(local_out, exist_ok=True)
        pull = subprocess.Popen(ssh + [f"cd {REMOTE} && cp run.log pip.log runs/{a.run}/ 2>/dev/null; "
                                       f"mkdir -p runs/{a.run}/gpu_profile && cp cache/*.json runs/{a.run}/gpu_profile/ 2>/dev/null; "
                                       f"cd runs/{a.run} && tar cf - --exclude=tmp_results ."], stdout=subprocess.PIPE)
        subprocess.run(["tar", "xf", "-", "-C", local_out], stdin=pull.stdout, check=True); pull.wait()
        print(f"  downloaded to {local_out}", flush=True)
        # keep this GPU model's engines for next time (gitignored; named by GPU, so they never clash with the laptop's)
        os.makedirs(ENGINES, exist_ok=True)
        pull_e = subprocess.Popen(ssh + [f"cd {REMOTE} && tar cf - {ENGINES} 2>/dev/null"], stdout=subprocess.PIPE)
        subprocess.run(["tar", "xf", "-", "-C", "."], stdin=pull_e.stdout); pull_e.wait()
        print(f"  saved {len(engines_for(offer.get('gpu_name')))} {offer.get('gpu_name')} engines to {ENGINES}/ for reuse", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        print(f"Destroying instance {iid} ...", flush=True)
        for _ in range(5):
            if subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True).returncode == 0:
                break
            time.sleep(5)
        try:
            cr.write_cloud_info(local_out, iid, offer, dph, t_start, time.time(), **remote_py)
        except Exception:
            pass
        hours = (time.time() - t_start) / 3600
        print(f"Done. ~{hours * 60:.0f} min, ~${hours * dph:.2f}. Check https://cloud.vast.ai/instances/ shows nothing left.", flush=True)
        summ = os.path.join(local_out, "summary.txt")
        if os.path.exists(summ):
            print("\n" + open(summ).read())


if __name__ == "__main__":
    main()
