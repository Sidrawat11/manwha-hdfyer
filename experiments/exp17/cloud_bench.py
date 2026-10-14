"""Exp 17 on a rented Vast.ai GPU: rent -> upload code + weights + benchmark chapters -> bench.py -> download -> destroy.

Reuses Exp 16's Vast plumbing (offer search, SSH, always-destroy). Uploads the code as committed
(git archive HEAD), the hdfyer weights, and the chapters in bench_chapters/ (~120 MB; they live only on the rented
machine and are deleted with it). TensorRT engines are NOT uploaded: they are tied to the GPU, so the server
builds its own and the warm-up runs report how long that takes.

Downloads only numbers and logs: results.json, summary.txt, logs/, dmon.log, plus the server's
cache/<gpu>.json (its measured batch profile) so the repo gains that GPU's profile.

Run from the repo root (WSL venv with vastai set up, as for Exp 16):
  python experiments/exp17/cloud_bench.py --dry-run
  python experiments/exp17/cloud_bench.py                       # H200 first, then H100s
  python experiments/exp17/cloud_bench.py --quick               # smoke test: batch 1 + auto only
  python experiments/exp17/cloud_bench.py --chapters-dir bench_chapters_par --parallel 1,2,4,6,8
"""
import argparse, os, shlex, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp16"))
import cloud_run as cr  # sh, vast, write_cloud_info

IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"   # base only; the laptop's torch 2.10 is installed on top
REMOTE = "/workspace/m"
WEIGHTS = "models/hdfyer/hdfyer-2x-s-r2.pth"
BENCH_DIR = "bench_chapters"  # default --chapters-dir; gitignored: the benchmark chapters as <series>/<chapter>.cbz


def bench_chapters(chapters_dir=BENCH_DIR):
    out = []
    for root, _, files in os.walk(chapters_dir):
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith(".cbz")]
    return sorted(out)


# Same versions as the laptop (requirements.txt). cu128 wheels need a driver that supports CUDA 12.8.
PIP = ("python -m pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 && "
       "python -m pip install -q spandrel==0.4.2 opencv-python-headless numpy imagesize tqdm "
       "tensorrt-cu12==11.4.0.106 onnx==1.23.2 onnxscript")


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
    ap.add_argument("--run", default=time.strftime("bench_%Y%m%d_%H%M"))
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--chapters-dir", default=BENCH_DIR, help="local folder of benchmark chapters (uploaded as bench_chapters/)")
    ap.add_argument("--parallel", default="", help='passed to bench.py, e.g. "1,2,4,6,8": the --parallel sweep only')
    a = ap.parse_args()

    CHAPTERS = bench_chapters(a.chapters_dir)
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
    local_out = os.path.join("experiments", "out", "exp17", a.run)
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

        print(f"  ssh ok; uploading code (git HEAD), weights and {len(CHAPTERS)} chapters...", flush=True)
        t = time.time()
        ssh_in = [x for x in ssh if x != "-n"]
        cr.sh(ssh + [f"mkdir -p {REMOTE}/bench_chapters"])
        git = subprocess.Popen(["git", "archive", "HEAD"], stdout=subprocess.PIPE)
        subprocess.run(ssh_in + [f"tar xf - -C {REMOTE}"], stdin=git.stdout, check=True); git.wait()
        tar = subprocess.Popen(["tar", "cf", "-", WEIGHTS, "experiments/exp17/bench.py"], stdout=subprocess.PIPE)  # bench.py: working copy, may be uncommitted
        subprocess.run(ssh_in + [f"tar xf - -C {REMOTE}"], stdin=tar.stdout, check=True); tar.wait()
        for c in CHAPTERS:  # flattened into bench_chapters/<series>/<chapter>.cbz
            dst = f"{REMOTE}/bench_chapters/{shlex.quote(os.path.basename(os.path.dirname(c)))}"
            with open(c, "rb") as f:
                subprocess.run(ssh_in + [f"mkdir -p {dst} && cat > {dst}/{shlex.quote(os.path.basename(c))}"], stdin=f, check=True)
        print(f"  uploaded in {time.time() - t:.0f}s", flush=True)

        bench_args = (["--chapters", "bench_chapters", "--out", f"runs/{a.run}"] + (["--quick"] if a.quick else [])
                      + (["--parallel", a.parallel] if a.parallel else []))
        # `&` binds to the WHOLE && chain, so the chain is one background shell; its own stdout/stdin must not be the
        # ssh channel, or ssh waits for it to finish (the "stuck at installing" hang of 12-13 Oct).
        remote_cmd = (f"cd {REMOTE} && setsid nohup bash -c \"({PIP}) > pip.log 2>&1 && "
                      f"python -c 'import torch, tensorrt, spandrel; print(torch.__version__, tensorrt.__version__, torch.cuda.get_device_name(0))' >> pip.log 2>&1 && "
                      f"PYTHONUNBUFFERED=1 python experiments/exp17/bench.py {' '.join(bench_args)} > run.log 2>&1\" "
                      f"> /dev/null 2>&1 < /dev/null &")
        print("  installing torch 2.10 + TensorRT 11.4 on the server (a few minutes)...", flush=True)
        try:
            subprocess.run(ssh + [remote_cmd], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            pass
        for _ in range(90):  # install runs in the background now: up to 15 min until bench.py appears
            if subprocess.run(ssh + ["pgrep -f exp17/bench.py"], capture_output=True).returncode == 0:
                break
            print("  ...installing: " + cr.sh(ssh + [f"du -sh /opt/conda/lib/python3*/site-packages 2>/dev/null | cut -f1"], check=False), flush=True)
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
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        print(f"Destroying instance {iid} ...", flush=True)
        for _ in range(5):
            if subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True).returncode == 0:
                break
            time.sleep(5)
        try:
            cr.write_cloud_info(local_out, iid, offer, dph, t_start, time.time())
        except Exception:
            pass
        hours = (time.time() - t_start) / 3600
        print(f"Done. ~{hours * 60:.0f} min, ~${hours * dph:.2f}. Check https://cloud.vast.ai/instances/ shows nothing left.", flush=True)
        summ = os.path.join(local_out, "summary.txt")
        if os.path.exists(summ):
            print("\n" + open(summ).read())


if __name__ == "__main__":
    main()
