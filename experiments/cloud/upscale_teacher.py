"""Upscales the training set and the exams with the teacher on a rented GPU and brings the PNGs home.

4x-UltraSharp, then a Lanczos half to 2x: the same load_model / run_model / half() / fp16 as distill.py's teacher
stage (the worker imports them from distill.py), so these targets match what training used.

    python experiments/cloud/upscale_teacher.py --smoke              # 10 pages, into data/teacher/ultrasharp-2x-smoke/
    python experiments/cloud/upscale_teacher.py --max-usd 10         # everything, into data/teacher/ultrasharp-2x/
    python experiments/cloud/upscale_teacher.py --dry-run            # plan, size estimate and disk check only; rents nothing

Output: <out>/training-v3/<series label>/<chapter>/<page>.png, <out>/exams/<label>/<page>.png, <out>/manifest.json.

Before renting it upscales 5 pages on this machine to estimate the output size, and refuses if free disk is below
1.5x that. Resumable: pages already downloaded are not sent again; the worker skips outputs it already has; a copy cut
off half-way is fetched again. --max-usd and --max-hours stop the upscaling early enough to download what is done.
The instance is destroyed at the end, on any error, and on Ctrl+C (the first Ctrl+C while upscaling stops the worker
and downloads the finished pages first; a second one destroys at once).
"""
import argparse, hashlib, io, json, os, shlex, shutil, signal, subprocess, sys, tarfile, tempfile, threading, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import vast
import teacher_jobs as tj
from outputs import output_dir

TRAIN, EXAMS = "data/training/v3", "data/exams"
MODEL = "models/4x-UltraSharp.pth"
DISTILL = "experiments/exp16/distill.py"
REMOTE = "/workspace/m"
SAMPLE_PAGES = 5


# ---------------------------------------------------------------- live view
class Live:
    """A few status lines redrawn in place: phase, progress bar with ETA, running spend. Log lines scroll above it."""

    def __init__(self, max_usd, max_hours):
        self.tty = sys.stdout.isatty()
        self.max_usd, self.max_hours = max_usd, max_hours
        self.phase, self.done, self.total = "", 0, 0
        self.phase_t0 = time.time()
        self.eta = None
        self.dph, self.gpu, self.t_bill, self.t_end = 0.0, "", None, None
        self.drawn, self.last_plain = 0, 0.0
        self.lock = threading.Lock()

    def spent(self):
        return tj.spend(self.dph, self.billed_s())

    def billed_s(self):
        return ((self.t_end or time.time()) - self.t_bill) if self.t_bill else 0.0

    def start(self, phase):
        """A new phase: its own elapsed time, bar and ETA."""
        self.phase, self.phase_t0, self.done, self.total, self.eta = phase, time.time(), 0, 0, None
        self.draw(force=True)

    def set(self, phase=None, done=None, total=None, eta=None):
        """Updates within the current phase (the text may change, e.g. the page number)."""
        for k, v in (("phase", phase), ("done", done), ("total", total), ("eta", eta)):
            if v is not None:
                setattr(self, k, v)
        self.draw()

    def lines(self):
        el = time.time() - self.phase_t0
        if self.total:
            eta = self.eta if self.eta is not None else tj.eta_s(self.done, self.total, el)
            prog = f"{tj.bar(self.done, self.total)}  ETA {tj.fmt_s(eta)}  elapsed {tj.fmt_s(el)}"
        else:
            prog = f"elapsed {tj.fmt_s(el)}"
        money = (f"Spend ${self.spent():.2f} of ${self.max_usd:.2f} cap" + (f"  ·  {self.gpu} at ${self.dph:.2f}/h" if self.gpu else "")
                 + f"  ·  {tj.fmt_s(self.billed_s())} of {tj.fmt_s(self.max_hours * 3600)}")
        return [self.phase, prog, money]

    def draw(self, force=False):
        with self.lock:
            lines = self.lines()
            if not self.tty:  # plain output (piped to a file): one line every 30 s
                if force or time.time() - self.last_plain >= 30:
                    print("  |  ".join(lines), flush=True); self.last_plain = time.time()
                return
            cols = shutil.get_terminal_size((100, 20)).columns - 1
            out = ("\x1b[F" * self.drawn) + "".join("\x1b[2K" + l[:cols] + "\n" for l in lines)
            sys.stdout.write(out); sys.stdout.flush()
            self.drawn = len(lines)

    def log(self, msg):
        with self.lock:
            if self.tty and self.drawn:
                sys.stdout.write("\x1b[F\x1b[2K" * self.drawn); self.drawn = 0
            print(msg, flush=True)
        self.draw(force=True)


def ticking(live, fn, every=1.0):
    """Runs fn() in a thread, redrawing the live view meanwhile (shows elapsed time during blocking calls)."""
    box = {}
    th = threading.Thread(target=lambda: box.update(r=_try(fn)), daemon=True)
    th.start()
    while th.is_alive():
        live.draw(); th.join(every)
    if "e" in box["r"]:
        raise box["r"]["e"]
    return box["r"]["v"]


def _try(fn):
    try:
        return {"v": fn()}
    except BaseException as e:
        return {"e": e}


# ---------------------------------------------------------------- local planning
def load_jobs():
    train = json.load(open(os.path.join(TRAIN, "manifest.json")))
    exams = {l: json.load(open(os.path.join(EXAMS, l, "manifest.json")))
             for l in sorted(os.listdir(EXAMS)) if os.path.isfile(os.path.join(EXAMS, l, "manifest.json"))}
    return tj.jobs_from_manifests(train, exams)


def local_src(src):
    top, rest = src.split("/", 1)
    return os.path.join(TRAIN if top == "training" else EXAMS, rest)


def local_sizes(out):
    sizes = {}
    for dirpath, _, files in os.walk(out):
        for f in files:
            if f.endswith(".png") and not f.endswith(".tmp.png"):
                p = os.path.join(dirpath, f)
                sizes[os.path.relpath(p, out).replace(os.sep, "/")] = os.path.getsize(p)
    return sizes


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_pixels(jobs):
    from PIL import Image
    px = []
    by_src = {}
    for j in jobs:
        by_src.setdefault(j["src"], []).append(j)
    for src, js in by_src.items():
        with zipfile.ZipFile(local_src(src)) as z:
            for j in js:
                w, h = Image.open(io.BytesIO(z.read(j["entry"]))).size
                px.append(w * h)
    return px


def sample_sizes(sample, model_sha, cache_path):
    """(input pixels, output PNG bytes) for the sample pages, upscaled here with the same code; cached by model + pages."""
    key = {"model_sha256": model_sha, "pages": [j["out"] for j in sample]}
    if os.path.exists(cache_path):
        c = json.load(open(cache_path))
        if c.get("key") == key:
            return [tuple(x) for x in c["sizes"]]
    import torch
    if not torch.cuda.is_available():
        os.environ["EXP16_DEVICE"] = "cpu"  # slow (minutes) but the size comes out the same
    sys.path.insert(0, os.path.dirname(os.path.join(ROOT, DISTILL)))
    import cv2, numpy as np
    import distill as d
    torch.backends.cudnn.benchmark = d.DEV == "cuda"
    teacher = d.load_model(os.path.join(ROOT, MODEL))
    sizes = []
    for k, j in enumerate(sample, 1):
        print(f"  sample page {k}/{len(sample)}", end="\r", flush=True)
        with zipfile.ZipFile(local_src(j["src"])) as z:
            img = cv2.imdecode(np.frombuffer(z.read(j["entry"]), np.uint8), cv2.IMREAD_COLOR)
        ok, png = cv2.imencode(".png", d.half(d.run_model(teacher, img)), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        sizes.append((img.shape[0] * img.shape[1], len(png)))
    print("\r" + " " * 40 + "\r", end="")
    del teacher; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump({"key": key, "sizes": sizes}, open(cache_path, "w"))
    return sizes


def upload_inputs(todo, tmp):
    """{arcname: local path} for the CBZs to upload. A CBZ of which only some pages are needed (smoke run, resume)
    is replaced by a small one holding just those entries, byte for byte."""
    files = {}
    for src in sorted({j["src"] for j in todo}):
        need = {j["entry"] for j in todo if j["src"] == src}
        with zipfile.ZipFile(local_src(src)) as z:
            if need >= {n for n in z.namelist()}:
                files[f"in/{src}"] = local_src(src); continue
            path = os.path.join(tmp, src.replace("/", "_"))
            with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as w:
                for n in sorted(need):
                    w.writestr(n, z.read(n))
        files[f"in/{src}"] = path
    return files


def free_bytes(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


# ---------------------------------------------------------------- transfers
class Counting(io.RawIOBase):
    """File wrapper counting bytes as they pass (for upload and download progress)."""

    def __init__(self, f, on_bytes, read=False):
        self.f, self.on_bytes, self.n, self._read = f, on_bytes, 0, read

    def readable(self):
        return self._read

    def writable(self):
        return not self._read

    def readinto(self, b):
        data = self.f.read(len(b))
        b[:len(data)] = data
        self.n += len(data); self.on_bytes(self.n)
        return len(data)

    def write(self, b):
        self.f.write(b)
        self.n += len(b); self.on_bytes(self.n)
        return len(b)


def upload(ssh, files, on_bytes):
    """Streams {arcname: local path} as one tar into REMOTE."""
    p = subprocess.Popen(vast.on_host(ssh, f"mkdir -p {REMOTE} && tar xf - -C {REMOTE}", stdin=True), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    with tarfile.open(fileobj=Counting(p.stdin, on_bytes), mode="w|") as tar:
        for arc, path in files.items():
            tar.add(path, arcname=arc)
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("upload failed: " + p.stderr.read().decode(errors="replace"))


def remote_sizes(ssh):
    out = vast.sh(vast.on_host(ssh, f"cd {REMOTE}/out 2>/dev/null && find . -name '*.png' ! -name '*.tmp.png' -printf '%s %P\\n'"), check=False)
    sizes = {}
    for line in out.splitlines():
        n, p = line.split(" ", 1)
        sizes[p] = int(n)
    return sizes


def download(ssh, paths, out, on_bytes):
    """Fetches the given output paths as a tar stream; each file is written to a temp name and renamed when complete."""
    subprocess.run(vast.on_host(ssh, f"cat > {REMOTE}/fetch.txt", stdin=True), input="\n".join(paths).encode(), check=True)
    p = subprocess.Popen(vast.on_host(ssh, f"cd {REMOTE}/out && tar cf - -T ../fetch.txt"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    got = 0
    try:
        with tarfile.open(fileobj=io.BufferedReader(Counting(p.stdout, on_bytes, read=True), 1 << 20), mode="r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                dst = os.path.join(out, *m.name.split("/"))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with tar.extractfile(m) as src, open(dst + ".part", "wb") as f:
                    shutil.copyfileobj(src, f, 1 << 20)
                os.replace(dst + ".part", dst); got += 1
    finally:
        if p.poll() is None:
            p.kill()
    if p.wait() != 0:
        raise RuntimeError("download failed: " + p.stderr.read().decode(errors="replace"))
    return got


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="10 pages spread over all sets")
    ap.add_argument("--out", help="default data/teacher/ultrasharp-2x (smoke: data/teacher/ultrasharp-2x-smoke)")
    ap.add_argument("--gpus", default="H200,H200_NVL,H100_SXM,H100_NVL", help="preference order (Vast gpu_name values)")
    ap.add_argument("--max-dph", type=float, default=4.5, help="price cap, $ per hour")
    ap.add_argument("--max-usd", type=float, default=10.0, help="hard spend cap for the whole job")
    ap.add_argument("--max-hours", type=float, default=3.0, help="hard time cap for the whole job")
    ap.add_argument("--geo", default="", help="country code filter, e.g. US")
    ap.add_argument("--batch", choices=["auto", "on", "off"], default="auto",
                    help="strip batching: auto = used only if within 1 level of the unbatched path on the first 5 pages")
    ap.add_argument("--image", default=vast.DEFAULT_IMAGE)
    ap.add_argument("--dry-run", action="store_true", help="plan, estimate and disk check; rent nothing")
    ap.add_argument("--yes", action="store_true", help="no confirmation prompt")
    a = ap.parse_args()
    os.chdir(ROOT)
    out = a.out or ("data/teacher/ultrasharp-2x-smoke" if a.smoke else "data/teacher/ultrasharp-2x")

    jobs = load_jobs()
    if a.smoke:
        jobs = tj.smoke_pick(jobs, 10)
    have = local_sizes(out)
    todo = tj.pending(jobs, have)
    print(f"{len(jobs)} pages in the job, {len(jobs) - len(todo)} already downloaded, {len(todo)} to do -> {out}/")
    if not todo:
        print("Nothing to do.")
        return

    model_sha = sha256(MODEL)
    print(f"Estimating output size from {SAMPLE_PAGES} pages upscaled here...", flush=True)
    sample = sample_sizes(tj.smoke_pick(jobs, SAMPLE_PAGES), model_sha, os.path.join(out, ".estimate.json"))
    est = tj.estimate_bytes(sample, input_pixels(todo))
    free = free_bytes(out)
    print(f"  expected download {tj.fmt_bytes(est)}; free here {tj.fmt_bytes(free)}")
    if not tj.disk_ok(free, est):
        sys.exit(f"Refusing: need at least 1.5x the expected size ({tj.fmt_bytes(1.5 * est)}) free at {out}.")

    upload_files = upload_inputs(todo, tempfile.mkdtemp(prefix="teacher-in-"))
    upload_files.update({"models/4x-UltraSharp.pth": MODEL, "code/distill.py": DISTILL,
                         "code/teacher_worker.py": os.path.join(HERE, "teacher_worker.py")})
    up_bytes = sum(os.path.getsize(p) for p in upload_files.values())
    disk = tj.remote_disk_gb(up_bytes, est)
    if a.dry_run:
        print(f"Dry run: would upload {tj.fmt_bytes(up_bytes)}, rent {disk} GB disk, caps ${a.max_usd:.2f} / {a.max_hours} h.")
        return

    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    offer = None
    for g in gpus:
        q = (f"gpu_name={g} num_gpus=1 verified=true rentable=true direct_port_count>=1 reliability>0.98 "
             f"inet_down>=300 inet_up>=200 disk_space>={disk} dph_total<={a.max_dph} cuda_max_good>=12.8 cpu_ram>=32"
             + (f" geolocation={a.geo}" if a.geo else ""))
        offers = vast.search(q)
        if offers:
            offer = offers[0]; break
        print(f"  no {g} under ${a.max_dph}/h right now")
    if not offer:
        sys.exit("No offer matched; try --max-dph or --gpus.")
    dph = float(offer["dph_total"])
    print(f"Offer: {offer['gpu_name']} at ${dph:.2f}/h in {offer.get('geolocation', '?')}, {disk} GB disk. "
          f"Caps: ${a.max_usd:.2f}, {a.max_hours} h (about {a.max_usd / dph:.1f} h at this price).")
    if not a.yes and input("Rent it? [y/N] ").strip().lower() != "y":
        return

    live = Live(a.max_usd, min(a.max_hours, 1e9))
    live.dph, live.gpu = dph, offer["gpu_name"]
    phases, run = {}, {"date": time.strftime("%Y-%m-%d %H:%M"), "smoke": a.smoke, "gpu": offer["gpu_name"], "dph": dph,
                       "geolocation": offer.get("geolocation"), "image": a.image, "pages_requested": len(todo)}
    t_phase = [time.time(), None]

    def phase(name, text):
        if t_phase[1]:
            phases[t_phase[1]] = round(time.time() - t_phase[0], 1)
        t_phase[:] = [time.time(), name]
        live.start(text)

    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.default_int_handler)  # even if started with SIGINT ignored
    signal.signal(signal.SIGTERM, interrupted)  # a closed terminal or kill also destroys the instance
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, interrupted)
    iid, ssh, ctl, stopped_by, error = None, None, "/tmp/vast-%C", None, None
    downloaded = 0
    try:
        phase("rent", f"Renting {offer['gpu_name']}…")
        iid = vast.rent(offer["id"], a.image, disk, "hdfyer-teacher")
        live.t_bill = time.time(); run["instance"] = iid
        live.log(f"Rented instance {iid}. It is destroyed when this script ends, however it ends.")
        phase("start", "Waiting for the instance to start…")
        ticking(live, lambda: vast.wait_running(iid, on_status=lambda s: live.set(f"Waiting for the instance to start… ({s or 'loading'})")))
        phase("ssh", "Waiting for SSH…")
        ssh = ticking(live, lambda: vast.ssh_command(iid, control_path=ctl))
        ticking(live, lambda: vast.wait_ssh(ssh))
        phase("python", "Finding the server's Python with torch…")
        py, torch_ver = ticking(live, lambda: vast.find_python(ssh))  # absolute path: a bare `python` may be the OS's
        run.update(python=py, torch=torch_ver)
        live.log(f"Server Python: {py} (torch {torch_ver}, CUDA available).")

        phase("upload", f"Uploading {tj.fmt_bytes(up_bytes)}")
        jobs_file = os.path.join(tempfile.mkdtemp(), "jobs.json")
        json.dump([{**j, "src": f"in/{j['src']}"} for j in todo], open(jobs_file, "w"))
        upload_files["jobs.json"] = jobs_file
        t_up = time.time()
        live.set(total=up_bytes)
        ticking(live, lambda: upload(ssh, upload_files, lambda n: setattr(live, "done", n)))
        up_rate = max(5e6, up_bytes / max(1.0, time.time() - t_up))  # download speed guess for the budget reserve

        phase("install", "Installing spandrel and OpenCV on the server")
        installed = ticking(live, lambda: vast.install_packages(ssh, py, torch_ver, log=f"{REMOTE}/pip.log"))  # checks torch is unchanged
        run.update(torchvision=installed["torchvision"])
        live.log(f"Installed; torch {installed['torch']} with CUDA unchanged, torchvision {installed['torchvision']}.")
        worker = (f"cd {REMOTE} && PYTHONUNBUFFERED=1 {shlex.quote(py)} code/teacher_worker.py --jobs jobs.json "
                  f"--model models/4x-UltraSharp.pth --out out --batch {a.batch} > worker.log 2>&1")
        subprocess.run(vast.on_host(ssh, f"rm -f {REMOTE}/progress.json && setsid bash -c {shlex.quote(worker)} > /dev/null 2>&1 < /dev/null &"),
                       timeout=60, check=True)

        phase("upscale", "Loading UltraSharp on the server…")
        poll = vast.on_host(ssh, f"cd {REMOTE}; cat progress.json 2>/dev/null; echo; pgrep -f '[t]eacher_worker.py' > /dev/null && echo ALIVE || echo DEAD")
        t_first, done_first, dead_polls, batch_note = None, 0, 0, None
        try:
            while True:
                r = subprocess.run(poll, capture_output=True, text=True, timeout=60)
                lines = r.stdout.strip().splitlines()
                alive = bool(lines) and lines[-1] == "ALIVE"
                try:
                    prog = json.loads(lines[0]) if len(lines) > 1 else None
                except json.JSONDecodeError:
                    prog = None
                if prog:
                    done, total = prog["done"], prog["total"]
                    if prog["phase"] in ("upscaling", "checking batching"):
                        if t_first is None and done > prog["skipped"]:
                            t_first, done_first = time.time(), done
                        eta = tj.eta_s(done - done_first, total - done_first, time.time() - t_first) if t_first else None
                        what = "checking strip batching on" if prog["phase"] == "checking batching" else "with UltraSharp"
                        live.set(f"Upscaling page {min(done + 1, total)}/{total} {what}" if done < total else "Upscaling done",
                                 done=done, total=total, eta=eta)
                    elif prog["phase"] == "loading model":
                        live.set("Loading UltraSharp on the server…")
                    b = prog.get("batch")
                    if b and "used" in b and batch_note is None:
                        batch_note = b
                        live.log(f"Strip batching {'on' if b['used'] else 'off'}: largest difference from the unbatched path "
                                 f"on {b['pages']} pages was {b['max_diff']} level(s).")
                    if prog["phase"] == "finished":
                        break
                    if prog["phase"] == "error":
                        error = "worker failed:\n" + prog.get("error", "")
                        live.log(error); break
                    remaining = max(0, total - done) * (est / max(1, len(todo)))
                    reserve_s = (est - remaining) / up_rate + 120  # time to download what is done, plus slack
                    if tj.over_budget(live.spent(), live.billed_s(), a.max_usd, a.max_hours, tj.spend(dph, reserve_s), reserve_s):
                        stopped_by = "budget"
                        live.log(f"Stopping the upscale at {done}/{total} pages: the spend or time cap is near. Downloading what is done.")
                        break
                if not alive:
                    dead_polls += 1
                    if dead_polls >= 3:
                        error = "worker died:\n" + vast.sh(vast.on_host(ssh, f"tail -n 30 {REMOTE}/worker.log"), check=False)
                        live.log(error); break
                else:
                    dead_polls = 0
                for _ in range(5):
                    live.draw(); time.sleep(1)
        except KeyboardInterrupt:
            stopped_by = "ctrl-c"
            live.log("Ctrl+C: stopping the worker and downloading the finished pages (Ctrl+C again destroys at once).")
        subprocess.run(vast.on_host(ssh, "pkill -f '[t]eacher_worker.py'"), capture_output=True, timeout=60)
        run["batching"] = batch_note

        phase("download", "Listing finished pages on the server…")
        rs = remote_sizes(ssh)
        want = tj.to_download(rs, local_sizes(out))
        total_b = sum(rs[p] for p in want)
        live.set(f"Downloading {len(want)} pages, {tj.fmt_bytes(total_b)}", total=total_b)

        def on_dl(n):
            live.done = n
            live.phase = f"Downloading {tj.fmt_bytes(n)}/{tj.fmt_bytes(total_b)}"
            if live.spent() >= a.max_usd or live.billed_s() >= a.max_hours * 3600:
                raise RuntimeError("spend or time cap reached during the download; rerun to fetch the rest")

        if want:
            downloaded = ticking(live, lambda: download(ssh, want, out, on_dl))
        still = tj.to_download(rs, local_sizes(out))
        if still:
            raise RuntimeError(f"{len(still)} files did not arrive intact; rerun to fetch them")
        run["pages_downloaded"], run["bytes_downloaded"] = downloaded, total_b
        left = tj.pending(todo, local_sizes(out))
        run["pages_left"] = len(left)
        phase(None, "Done" if not left else f"{len(left)} pages not done; rerun the same command to continue")
        if error:
            raise RuntimeError(error)
        if left and not stopped_by:
            raise RuntimeError(f"{len(left)} pages were not upscaled, for example {left[0]['out']}")
    except BaseException as e:
        error = error or f"{type(e).__name__}: {e}"
        raise
    finally:
        if t_phase[1]:
            phases[t_phase[1]] = round(time.time() - t_phase[0], 1)
        if iid:
            for sig in (signal.SIGINT, signal.SIGTERM) + ((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()):
                signal.signal(sig, signal.SIG_IGN)  # nothing may interrupt the destroy
            live.start(f"Destroying instance {iid}… (Ctrl+C is ignored until it is gone)")
            ok = vast.destroy(iid)
            live.t_end = time.time()  # the clock stops with the instance
            if ssh:
                subprocess.run(ssh[:-1] + ["-O", "exit", ssh[-1]], capture_output=True)
            minutes = live.billed_s() / 60
            run.update({"minutes": round(minutes, 1), "cost_usd": round(live.spent(), 2), "phase_seconds": phases,
                        "stopped_by": stopped_by, "error": error, "destroyed": ok})
            write_manifest(out, jobs, model_sha, run)
            live.start(f"Instance {iid} {'destroyed' if ok else 'NOT DESTROYED: run  vastai destroy instance ' + str(iid)}. "
                     f"{minutes:.0f} min, about ${live.spent():.2f}. Manifest: {out}/manifest.json")


def write_manifest(out, jobs, model_sha, run):
    path = os.path.join(out, "manifest.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    have = local_sizes(out)
    present = [j for j in jobs if have.get(j["out"], 0) > 0]
    runs = old.get("runs", []) + [run]
    output_dir(out, **{
        "model": {"file": MODEL, "sha256": model_sha, "bytes": os.path.getsize(MODEL)},
        "precision": "fp16: spandrel model .half(), input /255 in fp16, output clamped, rounded to uint8 (distill.py load_model/to_t/from_t)",
        "method": "4x model on 512-row strips with 32-row overlap, linear blend; then cv2 INTER_LANCZOS4 to (w//2, h//2); "
                  "PNG compression 1 (distill.py run_model/half)",
        "layout": "training-v3/<series label>/<chapter>/<page>.png, exams/<label>/<page>.png",
        "python": run.get("python") or old.get("python"), "torch": run.get("torch") or old.get("torch"),
        "torchvision": run.get("torchvision") or old.get("torchvision"),
        "page_counts_expected": tj.page_counts(jobs), "page_counts": tj.page_counts(present),
        "bytes": sum(have.get(j["out"], 0) for j in present),
        "minutes_total": round(sum(r.get("minutes", 0) for r in runs), 1),
        "cost_usd_total": round(sum(r.get("cost_usd", 0) for r in runs), 2),
        "runs": runs,
    })


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)  # the instance is already destroyed by then (see the finally in main)
