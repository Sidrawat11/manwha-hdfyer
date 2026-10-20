"""Upscales the training set and the exams with the teacher on a rented GPU and brings the PNGs home.

4x-UltraSharp, then a Lanczos half to 2x: the same load_model / run_model / half() / fp16 as distill.py's teacher
stage (the worker imports them from distill.py), so these targets match what training used.

    python experiments/cloud/upscale_teacher.py --smoke              # 10 pages, into data/teacher/ultrasharp-2x-smoke/
    python experiments/cloud/upscale_teacher.py --max-usd 10         # everything, into data/teacher/ultrasharp-2x/
    python experiments/cloud/upscale_teacher.py --dry-run            # plan, size estimate and disk check only; rents nothing
    python experiments/cloud/upscale_teacher.py --auto --max-usd 10 --deadline 23:30   # unattended, rotating bad hosts

Output: <out>/training-v3/<series label>/<chapter>/<page>.png, <out>/exams/<label>/<page>.png, <out>/manifest.json.

Before renting it upscales 5 pages on this machine to estimate the output size, and refuses if free disk is below
1.5x that. Resumable: pages already downloaded are not sent again; the worker skips outputs it already has; a copy cut
off half-way is fetched again. --max-usd and --max-hours stop the upscaling early enough to download what is done.
The instance is destroyed at the end, on any error, and on Ctrl+C (the first Ctrl+C while upscaling stops the worker
and downloads the finished pages first; a second one destroys at once).

--auto (unattended): no prompt; a bad host is destroyed, its machine added to data/teacher/.skip_machines.json (with
reason and time) and the next offer rented, up to --max-attempts. Bad = still loading after 6 min, ssh not up 3 min
after running, upload under --min-upload-mbps over its first 30 s, or upscale progress unchanged for 3 min (finished
pages are downloaded first). --max-usd and --max-hours cover all hosts together (hourly price plus per-GB traffic);
after --deadline no new host is rented and the current one downloads what is done. Locations matching --skip-geo
(default New Mexico) are never rented. <out>/status.log gets a line every minute and on every phase change or drop,
<out>/status.json the latest state; the last line is DONE or FAILED with a summary.
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
WORKER = os.path.join(HERE, "teacher_worker.py")
POLL_S = 5            # upscale progress poll
STATUS_EVERY_S = 60   # status.log line interval


# ---------------------------------------------------------------- live view
class Live:
    """Status lines redrawn in place: phase, progress bar with ETA, spend across all hosts, attempt and host.
    Log lines scroll above it. Also feeds the status log (status.log / status.json) once a minute and on events."""

    def __init__(self, max_usd, max_hours, max_attempts, t0=None):
        self.tty = sys.stdout.isatty()
        self.max_usd, self.max_hours, self.max_attempts = max_usd, max_hours, max_attempts
        self.t0 = time.time() if t0 is None else t0      # job start, for --max-hours
        self.phase, self.done, self.total, self.unit = "", 0, 0, None
        self.phase_t0 = time.time()
        self.eta = None
        self.dph, self.gpu, self.t_bill, self.t_end = 0.0, "", None, None
        self.attempt, self.offer, self.last_drop, self.spent_before = 0, None, None, 0.0
        self.sent = self.received = 0
        self.samples = []                                 # (time, bytes) for the transfer speed
        self.drawn, self.last_plain = 0, 0.0
        self.lock = threading.RLock()
        self.status = None                                # tj.StatusLog
        self.guard = None                                 # called while waiting; raises to stop the attempt

    def new_host(self, offer, attempt):
        self.offer, self.attempt = offer, attempt
        self.dph, self.gpu = float(offer["dph_total"]), offer.get("gpu_name", "")
        self.t_bill = self.t_end = None
        self.sent = self.received = 0  # bytes, for the host's per-GB traffic charges

    def traffic_usd(self, sent=None, received=None):
        """Per-GB charges: inet_down_cost for what we upload to the host, inet_up_cost for what we download."""
        o = self.offer or {}
        sent = self.sent if sent is None else sent
        received = self.received if received is None else received
        return sent / 1e9 * float(o.get("inet_down_cost") or 0) + received / 1e9 * float(o.get("inet_up_cost") or 0)

    def spent_current(self):
        return tj.spend(self.dph, self.billed_s()) + self.traffic_usd()

    def spent(self):
        """All hosts of this job so far, the current one included."""
        return self.spent_before + self.spent_current()

    def billed_s(self):
        return ((self.t_end or time.time()) - self.t_bill) if self.t_bill else 0.0

    def start(self, phase, unit=None):
        """A new phase: its own elapsed time, bar and ETA. Also a line in the status log."""
        self.phase, self.phase_t0, self.done, self.total, self.eta, self.unit = phase, time.time(), 0, 0, None, unit
        self.samples = []
        self.draw(force=True)
        if self.status:
            self.status.write()

    def set(self, phase=None, done=None, total=None, eta=None):
        """Updates within the current phase (the text may change, e.g. the page number)."""
        for k, v in (("phase", phase), ("done", done), ("total", total), ("eta", eta)):
            if v is not None:
                setattr(self, k, v)
        self.draw()

    def eta_now(self):
        if not self.total:
            return None
        return self.eta if self.eta is not None else tj.eta_s(self.done, self.total, time.time() - self.phase_t0)

    def speed_mb_s(self):
        """Transfer speed over the last ~10 s, for upload and download phases."""
        if self.unit != "bytes" or len(self.samples) < 2:
            return None
        (t0, b0), (t1, b1) = self.samples[0], self.samples[-1]
        return (b1 - b0) / 1e6 / (t1 - t0) if t1 > t0 else None

    def lines(self):
        el = time.time() - self.phase_t0
        if self.total:
            prog = f"{tj.bar(self.done, self.total)}  ETA {tj.fmt_s(self.eta_now())}  elapsed {tj.fmt_s(el)}"
            sp = self.speed_mb_s()
            prog += f"  {sp:.1f} MB/s" if sp is not None else ""
        else:
            prog = f"elapsed {tj.fmt_s(el)}"
        money = (f"Spend ${self.spent():.2f} of ${self.max_usd:.2f} cap (all hosts)" + (f"  ·  {self.gpu} at ${self.dph:.2f}/h" if self.gpu else "")
                 + f"  ·  {tj.fmt_s(time.time() - self.t0)} of {tj.fmt_s(self.max_hours * 3600)}")
        o = self.offer or {}
        host = (f"Attempt {self.attempt}/{self.max_attempts}  ·  machine {o.get('machine_id', '-')} ({o.get('geolocation') or '?'})"
                if self.offer else f"Attempt -/{self.max_attempts}")
        if self.last_drop:
            host += f"  ·  previous host {self.last_drop['machine_id']} dropped: {self.last_drop['reason']}"
        return [self.phase, prog, money, host]

    def draw(self, force=False):
        with self.lock:
            if self.unit == "bytes":
                now = time.time()
                self.samples.append((now, self.done))
                self.samples = [x for x in self.samples if now - x[0] <= 10]
            lines = self.lines()
            if self.tty:
                cols = shutil.get_terminal_size((100, 20)).columns - 1
                out = ("\x1b[F" * self.drawn) + "".join("\x1b[2K" + l[:cols] + "\n" for l in lines)
                sys.stdout.write(out); sys.stdout.flush()
                self.drawn = len(lines)
            elif force or time.time() - self.last_plain >= 30:  # plain output (piped to a file): one line every 30 s
                print("  |  ".join(lines), flush=True); self.last_plain = time.time()
            if self.status:
                self.status.tick()

    def log(self, msg):
        with self.lock:
            if self.tty and self.drawn:
                sys.stdout.write("\x1b[F\x1b[2K" * self.drawn); self.drawn = 0
            print(msg, flush=True)
        self.draw(force=True)


def ticking(live, fn, every=1.0):
    """Runs fn() in a thread, redrawing the live view meanwhile (shows elapsed time during blocking calls).
    live.guard(), if set, is called every second and may raise to abandon the wait (the thread is left to die with
    the instance)."""
    box = {}
    th = threading.Thread(target=lambda: box.update(r=_try(fn)), daemon=True)
    th.start()
    while th.is_alive():
        live.draw(); th.join(every)
        if live.guard and th.is_alive():
            live.guard()
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
        if self.f.closed:  # an aborted upload: tarfile flushes its buffer once more on cleanup
            return len(b)
        self.f.write(b)
        self.n += len(b); self.on_bytes(self.n)
        return len(b)


def upload(ssh, files, on_bytes):
    """Streams {arcname: local path} as one tar into REMOTE."""
    p = subprocess.Popen(vast.on_host(ssh, f"mkdir -p {REMOTE} && tar xf - -C {REMOTE}", stdin=True), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        tar = tarfile.open(fileobj=Counting(p.stdin, on_bytes), mode="w|")
        for arc, path in files.items():
            tar.add(path, arcname=arc)
        tar.close()
        p.stdin.close()
    except BaseException:
        p.kill()  # e.g. on_bytes judged the upload too slow
        raise
    if p.wait() != 0:
        raise RuntimeError("upload failed: " + p.stderr.read().decode(errors="replace"))


def remote_sizes(ssh):
    out = vast.sh(vast.on_host(ssh, f"cd {REMOTE}/out 2>/dev/null && find . -name '*.png' ! -name '*.tmp.png' -printf '%s %P\\n'"), check=False,
                  timeout=300)
    sizes = {}
    for line in out.splitlines():
        n, p = line.split(" ", 1)
        sizes[p] = int(n)
    return sizes


def download(ssh, paths, out, on_bytes):
    """Fetches the given output paths as a tar stream; each file is written to a temp name and renamed when complete."""
    subprocess.run(vast.on_host(ssh, f"cat > {REMOTE}/fetch.txt", stdin=True), input="\n".join(paths).encode(), check=True, timeout=120)
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
SKIP_FILE = "data/teacher/.skip_machines.json"  # machines dropped as bad hosts, kept across runs


class BadHost(Exception):
    """This host is unusable (stuck, unreachable, slow, stalled): destroy it, skip its machine, rent the next."""


class Stop(Exception):
    """A cap, the deadline or Ctrl+C: stop renting; what is downloaded stays."""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="10 pages spread over all sets")
    ap.add_argument("--out", help="default data/teacher/ultrasharp-2x (smoke: data/teacher/ultrasharp-2x-smoke)")
    ap.add_argument("--gpus", default="H200,H200_NVL,H100_SXM,H100_NVL", help="preference order (Vast gpu_name values)")
    ap.add_argument("--max-dph", type=float, default=4.5, help="price cap, $ per hour")
    ap.add_argument("--max-usd", type=float, default=10.0, help="hard spend cap for the whole job, all hosts together")
    ap.add_argument("--max-hours", type=float, default=3.0, help="hard time cap for the whole job")
    ap.add_argument("--deadline", help="HH:MM local time: no new host after it; the current one stops and downloads")
    ap.add_argument("--geo", default="", help="country code filter, e.g. US")
    ap.add_argument("--skip-machine", action="append", default=[], metavar="ID",
                    help=f"never rent this Vast machine id (repeatable); hosts dropped by --auto are added to {SKIP_FILE}")
    ap.add_argument("--skip-geo", action="append", default=["New Mexico"], metavar="TEXT",
                    help="never rent where the location contains this text (repeatable; default: New Mexico)")
    ap.add_argument("--auto", action="store_true", help="no prompt; a bad host is destroyed, skipped from then on, and the next offer rented")
    ap.add_argument("--max-attempts", type=int, default=5, help="hosts to try at most (--auto)")
    ap.add_argument("--min-upload-mbps", type=float, default=50, help="a host uploading slower (megabits/s, first 30 s) is bad")
    ap.add_argument("--batch", choices=["auto", "on", "off"], default="auto",
                    help="strip batching: auto = used only if within 1 level of the unbatched path on the first 5 pages")
    ap.add_argument("--image", default=vast.DEFAULT_IMAGE)
    ap.add_argument("--dry-run", action="store_true", help="plan, estimate and disk check; rent nothing")
    ap.add_argument("--yes", action="store_true", help="no confirmation prompt")
    a = ap.parse_args()
    os.chdir(ROOT)
    out = a.out or ("data/teacher/ultrasharp-2x-smoke" if a.smoke else "data/teacher/ultrasharp-2x")
    if not a.auto:
        a.max_attempts = 1

    jobs = load_jobs()
    if a.smoke:
        jobs = tj.smoke_pick(jobs, 10)
    todo = tj.pending(jobs, local_sizes(out))
    print(f"{len(jobs)} pages in the job, {len(jobs) - len(todo)} already downloaded, {len(todo)} to do -> {out}/")
    if not todo:
        print("Nothing to do.")
        return 0

    model_sha = sha256(MODEL)
    print(f"Estimating output size from {SAMPLE_PAGES} pages upscaled here...", flush=True)
    sample = sample_sizes(tj.smoke_pick(jobs, SAMPLE_PAGES), model_sha, os.path.join(out, ".estimate.json"))
    px = dict(zip((j["out"] for j in todo), input_pixels(todo)))
    est = tj.estimate_bytes(sample, px.values())
    free = free_bytes(out)
    print(f"  expected download {tj.fmt_bytes(est)}; free here {tj.fmt_bytes(free)}")
    if not tj.disk_ok(free, est):
        sys.exit(f"Refusing: need at least 1.5x the expected size ({tj.fmt_bytes(1.5 * est)}) free at {out}.")
    if a.dry_run:
        up = sum(os.path.getsize(p) for p in upload_inputs(todo, tempfile.mkdtemp(prefix="teacher-in-")).values()) + os.path.getsize(MODEL)
        print(f"Dry run: would upload {tj.fmt_bytes(up)}, rent {tj.remote_disk_gb(up, est)} GB disk, "
              f"caps ${a.max_usd:.2f} / {a.max_hours} h over at most {a.max_attempts} host(s).")
        return 0

    now = time.time()
    budget = tj.Budget(a.max_usd, a.max_hours, a.max_attempts, tj.parse_deadline(a.deadline, now), t0=now)
    live = Live(a.max_usd, a.max_hours, a.max_attempts, t0=now)
    J = argparse.Namespace(a=a, out=out, jobs=jobs, model_sha=model_sha, live=live, budget=budget, ctl="/tmp/vast-%C",
                           est_per_px=est / max(1, sum(px.values())), dl_rate=None)
    live.status = tj.StatusLog(out, lambda: snapshot(J), every=STATUS_EVERY_S)
    skips = {**{m: {"reason": "--skip-machine"} for m in a.skip_machine}, **tj.load_skips(SKIP_FILE)}

    signal.signal(signal.SIGINT, signal.default_int_handler)  # even if started with SIGINT ignored
    signal.signal(signal.SIGTERM, _interrupted)  # a closed terminal or kill also destroys the instance
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _interrupted)

    stop = None
    try:
        live.start("Planning…")
        while True:
            todo = tj.pending(jobs, local_sizes(out))
            if not todo:
                break
            why = budget.why_not_rent(0.0, time.time())
            if why:
                stop = why; break
            offer, disk = find_offer(J, todo, skips)
            if not offer:
                stop = "no offer matched (try --max-dph, --gpus or fewer skips)"; break
            why = budget.why_not_rent(float(offer["dph_total"]), time.time())
            if why:
                stop = why; break
            if not (a.auto or a.yes):
                print(f"Offer: {offer['gpu_name']} at ${float(offer['dph_total']):.2f}/h in {offer.get('geolocation', '?')} "
                      f"(machine {offer.get('machine_id')}), {disk} GB disk. Caps: ${a.max_usd:.2f}, {a.max_hours} h.")
                if input("Rent it? [y/N] ").strip().lower() != "y":
                    stop = "not rented"; break
            try:
                result = attempt(J, offer, len(budget.attempts) + 1, todo, disk)
            except BadHost as e:
                mid = offer.get("machine_id")
                reason = str(e)
                skips = tj.add_skip(SKIP_FILE, mid, reason, geolocation=offer.get("geolocation"), gpu=offer.get("gpu_name"),
                                    dph=float(offer["dph_total"]))
                live.last_drop = {"machine_id": mid, "reason": reason, "location": offer.get("geolocation")}
                live.log(f"Dropped machine {mid} ({offer.get('geolocation')}): {reason}. Skipped from now on ({SKIP_FILE}).")
                live.status.write()
                if not a.auto:
                    stop = f"bad host: {reason} (rerun with --auto to rotate hosts)"; break
                continue
            except Stop as e:
                stop = str(e); break
            if result != "done":
                stop = result; break
    except KeyboardInterrupt:
        stop = "interrupted (Ctrl+C)"
    finally:
        left = tj.pending(jobs, local_sizes(out))
        hosts = "; ".join(f"{x['machine_id']} ${x['cost_usd']:.2f}" + (f" dropped: {x['reason']}" if x["reason"] else "")
                          for x in budget.attempts) or "none"
        summary = (f"{len(jobs) - len(left)}/{len(jobs)} pages downloaded, ${budget.spent_before:.2f} spent over "
                   f"{len(budget.attempts)} host(s) [{hosts}]")
        final = ("DONE " if not left else f"FAILED ({stop or 'error'}) ") + summary
        live.offer, live.t_bill = None, None
        live.start(final)
        live.status.write(final=final)
    return 0 if not left else 1


def find_offer(J, todo, skips):
    """The cheapest acceptable offer for the preferred GPU types, and the disk to rent; searches again for up to
    10 min (--auto) when nothing matches."""
    a = J.a
    up_bytes = sum(os.path.getsize(local_src(s)) for s in {j["src"] for j in todo}) + os.path.getsize(MODEL)
    disk = tj.remote_disk_gb(up_bytes, J.est_per_px * sum(input_pixels(todo)))
    t0 = time.time()
    while True:
        for g in [g.strip() for g in a.gpus.split(",") if g.strip()]:
            q = (f"gpu_name={g} num_gpus=1 verified=true rentable=true direct_port_count>=1 reliability>0.98 "
                 f"inet_down>=300 inet_up>=200 disk_space>={disk} dph_total<={a.max_dph} cuda_max_good>=12.8 cpu_ram>=32"
                 + (f" geolocation={a.geo}" if a.geo else ""))
            offer = tj.pick_offer(vast.search(q), skips, a.skip_geo)
            if offer:
                return offer, disk
        if not a.auto or time.time() - t0 > 600:
            return None, disk
        J.live.set("No acceptable offer right now; searching again in 60 s…")
        for _ in range(60):
            J.live.draw(); time.sleep(1)


def snapshot(J):
    """The status.json contents (see tj.status_line for the log line)."""
    live, o = J.live, J.live.offer or {}
    left = tj.pending(J.jobs, local_sizes(J.out))
    return {"attempt": live.attempt, "attempts": J.a.max_attempts, "machine_id": o.get("machine_id"),
            "location": o.get("geolocation"), "gpu": o.get("gpu_name"), "dph": float(o["dph_total"]) if o else None,
            "phase": live.phase, "done": live.done, "total": live.total, "unit": live.unit,
            "speed_mb_s": live.speed_mb_s(), "eta_s": live.eta_now(), "spent_usd": round(live.spent(), 2),
            "max_usd": J.a.max_usd, "last_drop": live.last_drop, "pages_downloaded": len(J.jobs) - len(left),
            "pages_total": len(J.jobs), "hosts": J.budget.attempts}


def attempt(J, offer, n, todo, disk):
    """One host from rent to destroy. Returns "done" when no page is left, or the reason it stopped early
    (cap, deadline, Ctrl+C). Raises BadHost (rotate), Stop, or anything else (a real error)."""
    a, live, budget = J.a, J.live, J.budget
    upload_files = upload_inputs(todo, tempfile.mkdtemp(prefix="teacher-in-"))
    jobs_file = os.path.join(tempfile.mkdtemp(), "jobs.json")
    json.dump([{**j, "src": f"in/{j['src']}"} for j in todo], open(jobs_file, "w"))
    upload_files.update({"models/4x-UltraSharp.pth": MODEL, "code/distill.py": DISTILL,
                         "code/teacher_worker.py": WORKER, "jobs.json": jobs_file})
    up_bytes = sum(os.path.getsize(p) for p in upload_files.values())
    est = J.est_per_px * sum(input_pixels(todo))
    live.spent_before = budget.spent_before
    live.new_host(offer, n)
    dph, mid = live.dph, offer.get("machine_id")
    run = {"date": time.strftime("%Y-%m-%d %H:%M"), "attempt": n, "smoke": a.smoke, "gpu": offer.get("gpu_name"), "dph": dph,
           "geolocation": offer.get("geolocation"), "machine_id": mid, "image": a.image, "pages_requested": len(todo)}
    phases, t_phase = {}, [time.time(), None]

    def phase(name, text, unit=None):
        if t_phase[1]:
            phases[t_phase[1]] = round(time.time() - t_phase[0], 1)
        t_phase[:] = [time.time(), name]
        live.start(text, unit)

    def setup_guard():  # while setting up: any cap or the deadline ends the attempt (nothing to download yet)
        why = budget.must_stop(live.spent_current(), time.time())
        if why:
            raise Stop(why)

    iid, ssh, stopped_by, error, drop, outcome, caught = None, None, None, None, None, None, None
    try:
        live.guard = setup_guard
        phase("rent", f"Renting {offer.get('gpu_name')} on machine {mid} ({offer.get('geolocation')})…")
        iid = vast.rent(offer["id"], a.image, disk, "hdfyer-teacher")
        live.t_bill = time.time(); run["instance"] = iid
        live.log(f"Attempt {n}: rented instance {iid} on machine {mid} ({offer.get('geolocation')}, {offer.get('gpu_name')}, "
                 f"${dph:.2f}/h). It is destroyed when this attempt ends, however it ends.")
        phase("start", "Waiting for the instance to start…")
        try:
            ticking(live, lambda: vast.wait_running(iid, timeout_s=tj.LOADING_LIMIT_S,
                                                    on_status=lambda st: live.set(f"Waiting for the instance to start… ({st or 'loading'})")))
        except RuntimeError as e:
            if "not running after" in str(e):
                raise BadHost(f"still loading after {tj.fmt_s(tj.LOADING_LIMIT_S)}") from None
            raise
        phase("ssh", "Waiting for SSH…")
        plain = ticking(live, lambda: vast.ssh_command(iid))
        last = {"why": ""}

        def ssh_failed(k, why):
            last["why"] = why
            live.set(f"Waiting for SSH… attempt {k} failed: {why}")
        try:
            ticking(live, lambda: vast.wait_ssh(plain, tries=10 ** 6, max_s=tj.SSH_LIMIT_S, on_fail=ssh_failed))
        except RuntimeError:
            raise BadHost(f"ssh not up {tj.fmt_s(tj.SSH_LIMIT_S)} after running ({last['why'] or 'no answer'})") from None
        ssh = vast.with_control(plain, J.ctl)  # multiplexed only now that a connection is known to work
        phase("python", "Finding the server's Python with torch…")
        try:
            py, torch_ver = ticking(live, lambda: vast.find_python(ssh))
        except subprocess.TimeoutExpired:
            raise BadHost("the Python probe timed out") from None
        run.update(python=py, torch=torch_ver)
        live.log(f"Server Python: {py} (torch {torch_ver}, CUDA available).")

        phase("upload", f"Uploading {tj.fmt_bytes(up_bytes)}", unit="bytes")
        live.set(total=up_bytes)
        t_up, judged = time.time(), [False]

        def on_up(nb):
            live.done = live.sent = nb
            if not judged[0]:
                slow = tj.upload_too_slow(nb, time.time() - t_up, a.min_upload_mbps)
                if slow is not None:
                    judged[0] = True
                    if slow:
                        raise BadHost(f"upload {nb * 8 / (time.time() - t_up) / 1e6:.0f} Mbit/s over its first "
                                      f"{tj.UPLOAD_WINDOW_S} s, under --min-upload-mbps {a.min_upload_mbps:g}")
        try:
            ticking(live, lambda: upload(ssh, upload_files, on_up))
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
            raise BadHost(f"upload failed: {str(e).strip()[:200]}") from None
        up_rate = up_bytes / max(1.0, time.time() - t_up)
        run["upload_mb_s"] = round(up_rate / 1e6, 1)
        live.log(f"Uploaded {tj.fmt_bytes(up_bytes)} at {up_rate / 1e6:.1f} MB/s.")
        dl_rate = J.dl_rate or max(5e6, up_rate)  # download speed guess for the budget reserve until one is measured

        phase("install", "Installing spandrel and OpenCV on the server")
        try:
            installed = ticking(live, lambda: vast.install_packages(ssh, py, torch_ver, log=f"{REMOTE}/pip.log"))
        except subprocess.TimeoutExpired:
            raise BadHost("pip install timed out") from None
        except RuntimeError as e:
            if "pip install failed" in str(e):
                raise BadHost(f"pip install failed: {str(e).strip().splitlines()[-1][:200]}") from None
            raise
        run.update(torchvision=installed["torchvision"])
        live.log(f"Installed; torch {installed['torch']} with CUDA unchanged, torchvision {installed['torchvision']}.")
        worker = (f"PYTHONUNBUFFERED=1 {shlex.quote(py)} code/teacher_worker.py --jobs jobs.json "
                  f"--model models/4x-UltraSharp.pth --out out --batch {a.batch} > worker.log 2>&1")
        try:
            vast.start_detached(ssh, worker, "teacher_worker.py", cwd=REMOTE, before=f"rm -f {REMOTE}/progress.json",
                                log=f"{REMOTE}/worker.log")
        except subprocess.TimeoutExpired:
            raise BadHost("starting the worker timed out") from None

        live.guard = None
        phase("upscale", "Loading UltraSharp on the server…", unit="pages")
        poll = vast.on_host(ssh, f"cd {REMOTE}; cat progress.json 2>/dev/null; echo; "
                                 f"{vast.pgrep_command('teacher_worker.py')} > /dev/null && echo ALIVE || echo DEAD")
        t_first, done_first, dead_polls, batch_note = None, 0, 0, None
        last_key, last_change = None, time.time()
        try:
            while True:
                try:
                    r = subprocess.run(poll, capture_output=True, text=True, timeout=60)
                    lines = r.stdout.strip().splitlines()
                except subprocess.TimeoutExpired:
                    lines = []
                alive = bool(lines) and lines[-1] == "ALIVE"
                try:
                    prog = json.loads(lines[0]) if len(lines) > 1 else None
                except json.JSONDecodeError:
                    prog = None
                if prog:
                    done, total = prog["done"], prog["total"]
                    if (prog["phase"], done) != last_key:
                        last_key, last_change = (prog["phase"], done), time.time()
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
                    done_est = est * (done - prog["skipped"]) / max(1, total - prog["skipped"])
                    reserve_s = done_est / dl_rate + 120  # time to download what is done, plus slack
                    reserve_usd = tj.spend(dph, reserve_s) + live.traffic_usd(0, done_est)
                    why = budget.must_stop(live.spent_current(), time.time(), reserve_usd, reserve_s)
                    if why:
                        stopped_by = why
                        live.log(f"Stopping the upscale at {done}/{total} pages: {why} is near. Downloading what is done.")
                        break
                if time.time() - last_change > tj.STALL_LIMIT_S:
                    where = f" at page {last_key[1]}" if last_key else " before the worker reported anything"
                    drop = f"upscale stalled for {tj.fmt_s(tj.STALL_LIMIT_S)}{where}"
                    live.log(f"{drop}; downloading what is done, then dropping this host."); break
                if not alive and lines:
                    dead_polls += 1
                    if dead_polls >= 3:
                        tail = vast.sh(vast.on_host(ssh, f"tail -n 30 {REMOTE}/worker.log"), check=False, timeout=60)
                        drop = "worker vanished: " + (tail.strip().splitlines() or ["no log"])[-1][:200]
                        live.log(f"{drop}; downloading what is done, then dropping this host."); break
                else:
                    dead_polls = 0
                for _ in range(max(1, round(POLL_S / 0.2))):
                    live.draw(); time.sleep(0.2)
        except KeyboardInterrupt:
            stopped_by = "Ctrl+C"
            live.log("Ctrl+C: stopping the worker and downloading the finished pages (Ctrl+C again destroys at once).")
        try:
            subprocess.run(vast.on_host(ssh, "pkill -f '[t]eacher_worker.py'"), capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            pass
        run["batching"] = batch_note

        def dl_guard():  # downloading: only the hard spend cap stops it
            if budget.spent_before + live.spent_current() >= budget.cap():
                raise Stop("budget reached during the download")
        live.guard = dl_guard
        got, moved, last_err, t_dl = 0, 0, None, time.time()
        for k in range(3):
            phase("download", "Listing finished pages on the server…", unit="bytes")
            try:
                rs = remote_sizes(ssh)
                want = tj.to_download(rs, local_sizes(J.out))
                total_b = sum(rs[p] for p in want)
                live.set(f"Downloading {len(want)} pages, {tj.fmt_bytes(total_b)}", total=total_b)

                def on_dl(nb, base=live.received):
                    live.done, live.received = nb, base + nb
                    live.phase = f"Downloading {tj.fmt_bytes(nb)}/{tj.fmt_bytes(total_b)}"
                if want:
                    got += ticking(live, lambda: download(ssh, want, J.out, on_dl))
                    moved += total_b
                if not tj.to_download(rs, local_sizes(J.out)):
                    last_err = None; break
                last_err = "files did not arrive intact"
            except (RuntimeError, subprocess.TimeoutExpired, OSError, tarfile.TarError) as e:
                last_err = str(e).strip()[:200]
                live.log(f"Download try {k + 1} failed: {last_err}")
        if moved and time.time() > t_dl:
            J.dl_rate = moved / (time.time() - t_dl)
            run["download_mb_s"] = round(J.dl_rate / 1e6, 1)
        run["pages_downloaded"], run["bytes_downloaded"] = got, moved
        if last_err:
            raise BadHost(f"download failed 3 times: {last_err}")
        left = tj.pending(J.jobs, local_sizes(J.out))
        run["pages_left"] = len(left)
        if error:
            raise RuntimeError(error)
        if drop:
            raise BadHost(drop)
        outcome = "done" if not left else (stopped_by or f"{len(left)} pages not upscaled")
        return outcome
    except BaseException as e:
        error, caught = error or f"{type(e).__name__}: {e}", e
        raise
    finally:
        live.guard = None
        if t_phase[1]:
            phases[t_phase[1]] = round(time.time() - t_phase[0], 1)
        reason = str(caught) if isinstance(caught, BadHost) else None
        if iid:
            for sig in (signal.SIGINT, signal.SIGTERM) + ((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()):
                signal.signal(sig, signal.SIG_IGN)  # nothing may interrupt the destroy
            live.start(f"Destroying instance {iid}… (Ctrl+C is ignored until it is gone)")
            ok = vast.destroy(iid)
            live.t_end = time.time()  # the clock stops with the instance
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.signal(signal.SIGTERM, _interrupted)
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, _interrupted)
            if ssh:  # close the multiplexing master
                try:
                    subprocess.run(ssh[:-1] + ["-O", "exit", ssh[-1]], capture_output=True, timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            cost = live.spent_current()
            run.update({"minutes": round(live.billed_s() / 60, 1), "cost_usd": round(cost, 2), "phase_seconds": phases,
                        "stopped_by": stopped_by, "dropped": reason, "error": None if reason else error, "destroyed": ok})
            budget.record(mid, cost, reason)
            live.spent_before = budget.spent_before
            live.t_bill = None
            write_manifest(J.out, J.jobs, J.model_sha, run)
            live.log(f"Instance {iid} {'destroyed' if ok else 'NOT DESTROYED: run  vastai destroy instance ' + str(iid)}. "
                     f"This host: {run['minutes']:.0f} min, about ${cost:.2f}; all hosts ${budget.spent_before:.2f}.")
            if not ok:
                raise RuntimeError(f"instance {iid} could not be destroyed: run  vastai destroy instance {iid}")
        else:
            budget.record(mid, 0.0, reason)


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


def _interrupted(*_):
    raise KeyboardInterrupt


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)  # the instance is already destroyed by then (see the finally in attempt)
