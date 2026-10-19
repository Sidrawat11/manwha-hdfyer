"""Exp 16 on a rented Vast.ai GPU, end to end: rent -> upload -> train -> download -> destroy.

What it does
  1. Finds the cheapest suitable offer, trying GPUs in your order of preference, under a price cap.
  2. Shows you the GPU, price and worst-case cost, and asks y/N (skip with --yes).
  3. Rents it, waits until it is up, uploads only what is needed (the dataset in --data, e.g. data/training/v3,
     the two model files, the exp16 + exp15 scripts).
  4. Runs `distill.py all` there in the background and streams its progress here.
  5. Downloads results (checkpoints, logs, comparers, summary) to outputs/exp16-cloud/<run>/.
     Teacher PNGs and the patch bank stay on the server (gigabytes, cheap to regenerate).
  6. ALWAYS destroys the instance at the end: on success, error, Ctrl+C, or the --max-hours limit.
     Destroying (not stopping) is what ends all billing, storage included.

One-time setup (in your WSL venv):
  pip install vastai
  vastai set api-key <key from https://console.vast.ai/manage-keys/>
  ssh-keygen -t ed25519            (only if ~/.ssh/id_ed25519 doesn't exist yet; just press Enter)
  vastai create ssh-key ~/.ssh/id_ed25519.pub

Run from the repo root:
  python experiments/exp16/cloud_run.py --dry-run          # just show what it would rent and the cost
  python experiments/exp16/cloud_run.py                    # do it (asks before renting)
  python experiments/exp16/cloud_run.py --steps 60000 --max-dph 1.5 --gpus H100_PCIE,RTX_4090
Anything after `--` is passed to distill.py, e.g.  -- --batch 64 --lr 3e-4

If this script is killed so hard that it can't clean up (laptop dies), check https://cloud.vast.ai/instances/
or run `vastai show instances` and `vastai destroy instance <id> -y`. (A container can't reliably
switch its own billing off, so the --max-hours limit is enforced by this script, not by the server.)
"""
import argparse, json, os, shlex, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cloud"))
from vast import find_python, sh, vast, write_cloud_info  # shared Vast.ai helpers

IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
REMOTE = "/workspace/m"
INIT = "models/2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.safetensors"
TEACHER = "models/4x-UltraSharp.pth"
UPLOAD = [TEACHER, INIT, "experiments/exp16", "experiments/exp15_native2x.py",
          "experiments/common.py", "experiments/exp06_split_vs_full.py"]


def find_offer(gpus, max_dph, disk, geo=""):
    for g in gpus:
        q = (f"gpu_name={g} num_gpus=1 verified=true rentable=true direct_port_count>=1 reliability>0.98 "
             f"inet_down>=300 inet_up>=100 disk_space>={disk} dph_total<={max_dph} cuda_max_good>=12.4 cpu_ram>=32"
             + (f" geolocation={geo}" if geo else ""))
        offers = vast("search", "offers", q, "-o", "dph_total", check=False)
        if isinstance(offers, list) and offers:
            return offers[0]
        print(f"  no {g} under ${max_dph}/h right now", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="H100_PCIE,H100_SXM,H100_NVL,A100_SXM4,RTX_5090,RTX_4090",
                    help="preference order (Vast gpu_name values)")
    ap.add_argument("--max-dph", type=float, default=2.5, help="price cap, $ per hour")
    ap.add_argument("--geo", default="", help="country code filter, e.g. US")
    ap.add_argument("--max-hours", type=float, default=2.0, help="hard limit; destroyed after this")
    ap.add_argument("--disk", type=int, default=50, help="GB")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--run", default=time.strftime("run_%Y%m%d_%H%M"))
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data", default="data/training/v3", help="dataset folder to upload and train on")
    ap.add_argument("--attach", type=int, default=0,
                    help="take over an instance that is already training: watch, download, destroy")
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]

    UPLOAD.insert(0, a.data)
    for p in UPLOAD:
        if not os.path.exists(p):
            sys.exit(f"missing {p} (run from the repo root; build the dataset with experiments/data/rebuild_datasets.py)")
    try:
        user = vast("show", "user")
        bal = user.get("credit", user.get("balance")) if isinstance(user, dict) else None
    except Exception as e:
        sys.exit(f"vastai not ready ({e}). See the setup steps at the top of this file.")
    print(f"Vast.ai credit: ${bal}", flush=True)

    if a.attach:
        info = vast("show", "instance", str(a.attach), check=False)
        if not isinstance(info, dict) or not info.get("id"):
            sys.exit(f"instance {a.attach} not found")
        offer = {"id": None, "dph_total": info.get("dph_total", 0), "gpu_name": info.get("gpu_name"),
                 "gpu_ram": info.get("gpu_ram", 0), "cpu_ram": info.get("cpu_ram", 0),
                 "geolocation": info.get("geolocation", "?"), "reliability2": info.get("reliability2", "?")}
        a.yes = True
    else:
        offer = find_offer([g.strip() for g in a.gpus.split(",")], a.max_dph, a.disk, a.geo)
    if not offer:
        sys.exit("No offer matched. Raise --max-dph or add GPUs to --gpus.")
    dph = float(offer["dph_total"])
    print(f"Best offer: {offer.get('gpu_name')} {offer.get('gpu_ram', 0) / 1024:.0f} GB GPU memory | "
          f"{offer.get('cpu_ram', 0) / 1024:.0f} GB RAM | ${dph:.3f}/h | {offer.get('geolocation', '?')} | "
          f"reliability {offer.get('reliability2', offer.get('reliability', '?'))} | offer {offer['id']}", flush=True)
    print(f"Worst case at --max-hours {a.max_hours}: ${dph * a.max_hours:.2f} (+ small bandwidth/storage)", flush=True)
    if a.dry_run:
        return
    if bal is not None and float(bal) < dph * a.max_hours:
        print(f"WARNING: credit ${bal} is below the worst case.", flush=True)
    if not a.yes and input("Rent it? [y/N] ").strip().lower() != "y":
        return

    if a.attach:
        iid = a.attach
        print(f"Attached to instance {iid}. It will be destroyed when this script ends.", flush=True)
    else:
        res = vast("create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(a.disk), "--ssh", "--direct",
                   "--label", f"manwha-exp16-{a.run}")
        iid = res.get("new_contract") if isinstance(res, dict) else None
        if not iid:
            sys.exit(f"create failed: {res}")
        print(f"Rented instance {iid}. It will be destroyed when this script ends.", flush=True)
    t_start = time.time(); deadline = t_start + a.max_hours * 3600
    local_out = os.path.join("outputs", "exp16-cloud", a.run)
    remote_py = {}  # the server's Python with torch, recorded in cloud.json
    try:
        # wait for running + ssh
        while True:
            info = vast("show", "instance", str(iid), check=False)
            st = info.get("actual_status") if isinstance(info, dict) else None
            print(f"  status: {st}", flush=True)
            if st == "running":
                break
            if time.time() - t_start > 900:
                raise RuntimeError("instance not running after 15 min")
            time.sleep(15)
        url = sh(["vastai", "ssh-url", str(iid)])  # ssh://root@host:port
        hostport = url.split("://", 1)[1]
        userhost, port = hostport.rsplit(":", 1)
        ssh = ["ssh", "-n", "-p", port, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", userhost]
        for _ in range(40):
            if subprocess.run(ssh + ["true"], capture_output=True).returncode == 0:
                break
            time.sleep(10)
        else:
            raise RuntimeError("ssh never came up")
        py, torch_ver = find_python(ssh)  # absolute path: a bare `python` over ssh may be the OS's, without torch
        remote_py.update(python=py, torch=torch_ver)
        print(f"  server Python: {py} (torch {torch_ver})", flush=True)
        if a.attach:
            a.run = sh(ssh + [f"ls -t {REMOTE}/runs | head -1"])
            local_out = os.path.join("outputs", "exp16-cloud", a.run)
            print(f"  watching run {a.run}", flush=True)
        if not a.attach:
            upload_and_start(a, ssh, extra, py)
        watch_download(a, ssh, iid, t_start, deadline, dph, local_out)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        print(f"Destroying instance {iid} ...", flush=True)
        for _ in range(5):
            r = subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True, text=True)
            if r.returncode == 0:
                break
            time.sleep(5)
        try:
            write_cloud_info(local_out, iid, offer, dph, t_start, time.time(), **remote_py)
        except Exception:
            pass
        hours = (time.time() - t_start) / 3600
        print(f"Done. ~{hours * 60:.0f} min on this script's watch, ~${hours * dph:.2f}. "
              f"Check https://cloud.vast.ai/instances/ shows nothing left.", flush=True)
        summ = os.path.join(local_out, "summary.txt")
        if os.path.exists(summ):
            print("\n" + open(summ).read())


def upload_and_start(a, ssh, extra, py):
        print("  ssh ok; uploading (~340 MB)...", flush=True)
        t = time.time()
        tar = subprocess.Popen(["tar", "cf", "-", *UPLOAD], stdout=subprocess.PIPE)
        subprocess.run([x for x in ssh if x != "-n"] + [f"mkdir -p {REMOTE} && tar xf - -C {REMOTE}"], stdin=tar.stdout, check=True)
        tar.wait()
        print(f"  uploaded in {time.time() - t:.0f}s", flush=True)

        run_args = ["--data", a.data, "--out", f"runs/{a.run}", "--steps", str(a.steps), *extra]
        # --no-deps: never let pip replace the image's PyTorch (a multi-GB download); add only the small pieces
        # NOTE: `&` below binds to the whole && chain, so that chain is one background shell holding the ssh channel:
        # that was the "stuck at installing packages" hang (ssh waited for it). The timeout + pgrep below copes with
        # it; experiments/exp17/cloud_bench.py has the clean fix (setsid bash -c "..." > /dev/null 2>&1 < /dev/null &).
        remote_cmd = (f"cd {REMOTE} && {py} -m pip install --no-deps spandrel==0.4.2 einops safetensors opencv-python-headless tqdm "
                      f"> pip.log 2>&1 && {py} -c 'import spandrel, cv2, torch, torchvision; print(torch.__version__)' && "
                      f"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
                      f"PYTHONUNBUFFERED=1 setsid nohup {py} experiments/exp16/distill.py all {' '.join(shlex.quote(x) for x in run_args)} "
                      f"> run.log 2>&1 < /dev/null &")  # </dev/null: otherwise ssh waits for the background job and never returns
        print("  installing packages on the server...", flush=True)
        try:  # the background job can keep the ssh channel open; a timeout here is fine if training is running
            subprocess.run(ssh + [remote_cmd], capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            pass
        for _ in range(12):
            if subprocess.run(ssh + ["pgrep -f distill.py"], capture_output=True).returncode == 0:
                break
            time.sleep(10)
        else:
            raise RuntimeError("training did not start; see pip.log / run.log on the server")


def watch_download(a, ssh, iid, t_start, deadline, dph, local_out):
        print("  training running; progress every 30 s (Ctrl+C stops and destroys the instance)", flush=True)
        last = ""
        while True:
            time.sleep(30)
            out = subprocess.run(ssh + [f"cd {REMOTE}; tr '\\r' '\\n' < run.log | grep -v '^\\s*$' | tail -n 4; "
                                        f"test -f runs/{a.run}/DONE && echo __DONE__; "
                                        f"pgrep -f distill.py >/dev/null || echo __DEAD__"],
                                 capture_output=True, text=True).stdout
            lines = [l for l in out.splitlines() if not l.startswith("__")]
            if lines and lines[-1] != last:
                last = lines[-1]
                spent = (time.time() - t_start) / 3600 * dph
                print(f"  [{(time.time() - t_start) / 60:5.1f} min, ~${spent:.2f}] {last[:160]}", flush=True)
            if "__DONE__" in out:
                break
            if "__DEAD__" in out:
                tail = sh(ssh + [f"tail -n 40 {REMOTE}/run.log"], check=False)
                raise RuntimeError(f"distill.py stopped without finishing:\n{tail}")
            if time.time() > deadline - 600:
                print("  approaching --max-hours: downloading what exists, then destroying", flush=True)
                break

        os.makedirs(local_out, exist_ok=True)
        print(f"  downloading results to {local_out} ...", flush=True)
        pull = subprocess.Popen(ssh + [f"cd {REMOTE}/runs/{a.run} && tar cf - --exclude=teacher --exclude=bank "
                                       f"--exclude=last.pt . ; cp ../../run.log . 2>/dev/null"], stdout=subprocess.PIPE)
        subprocess.run(["tar", "xf", "-", "-C", local_out], stdin=pull.stdout, check=True)
        pull.wait()
        subprocess.run(ssh + [f"cat {REMOTE}/run.log"], stdout=open(os.path.join(local_out, "run.log"), "w"))
        print("  downloaded.", flush=True)


if __name__ == "__main__":
    main()
