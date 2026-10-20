"""Pure parts of the teacher upscale job: which pages, where their outputs go, what is left to do, how big it gets.

No GPU, no network; tests in tests/test_cloud.py.
"""
import json, math, os, time

TRAIN_PREFIX, EXAM_PREFIX = "training-v3", "exams"


def stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def jobs_from_manifests(train_manifest, exam_manifests):
    """One job per page: {"src": cbz path relative to the data root, "entry": name inside it, "out": output path}.

    Training pages (train_*.cbz and test.cbz) go to training-v3/<series label>/<chapter>/<page>.png, using the
    page's original chapter and page name; exam pages go to exams/<label>/<entry>.png. src is relative to
    data/training/v3 for training jobs ("training/" prefix) and to data/exams for exam jobs ("exams/" prefix).
    """
    jobs = []
    sets = list(train_manifest["train"].items()) + [("test.cbz", train_manifest["test"])]
    for cbz, entries in sets:
        for e in entries:
            jobs.append({"src": f"training/{cbz}", "entry": e["entry"],
                         "out": f"{TRAIN_PREFIX}/{e['series_label']}/{stem(e['cbz'])}/{stem(e['page'])}.png"})
    for label in sorted(exam_manifests):
        for e in exam_manifests[label]["test"]:
            jobs.append({"src": f"exams/{label}/test.cbz", "entry": e["entry"], "out": f"{EXAM_PREFIX}/{label}/{stem(e['entry'])}.png"})
    outs = [j["out"] for j in jobs]
    dup = sorted({o for o in outs if outs.count(o) > 1})
    if dup:
        raise ValueError(f"two pages map to the same output: {dup[:3]}")
    return jobs


def page_counts(jobs):
    """{"training-v3": {"L1": n, ...}, "exams": {"A": n, ...}}"""
    counts = {}
    for j in jobs:
        top, label = j["out"].split("/")[:2]
        counts.setdefault(top, {}).setdefault(label, 0)
        counts[top][label] += 1
    return counts


def pending(jobs, local_sizes):
    """Jobs whose output is not already downloaded (a zero-byte file counts as missing)."""
    return [j for j in jobs if local_sizes.get(j["out"], 0) <= 0]


def to_download(remote_sizes, local_sizes):
    """Remote outputs that are missing locally or differ in size (an interrupted copy), in path order."""
    return sorted(p for p, n in remote_sizes.items() if local_sizes.get(p) != n)


def smoke_pick(jobs, n):
    """n jobs spread evenly over the list, so a smoke run touches every set."""
    if n >= len(jobs):
        return list(jobs)
    return [jobs[round(k * (len(jobs) - 1) / (n - 1))] for k in range(n)] if n > 1 else jobs[:1]


def estimate_bytes(sample, input_pixels):
    """Output size for pages with the given input pixel counts, from sample (input pixels, output PNG bytes) pairs."""
    px = sum(p for p, _ in sample)
    if not px:
        raise ValueError("empty sample")
    per_px = sum(b for _, b in sample) / px
    return int(math.ceil(per_px * sum(input_pixels)))


def disk_ok(free_bytes, need_bytes, factor=1.5):
    return free_bytes >= factor * need_bytes


def remote_disk_gb(input_bytes, output_bytes):
    """Instance disk: inputs + 1.5x the expected outputs + room for the image and pip, at least 40 GB."""
    return max(40, math.ceil((input_bytes + 1.5 * output_bytes) / 1e9) + 20)


def spend(dph, seconds):
    return dph * seconds / 3600


def over_budget(spent_usd, elapsed_s, max_usd, max_hours, reserve_usd=0.0, reserve_s=0.0):
    """True once the spend or time (plus what is still needed to download) reaches a cap."""
    return spent_usd + reserve_usd >= max_usd or elapsed_s + reserve_s >= max_hours * 3600


def eta_s(done, total, elapsed_s):
    if done <= 0 or elapsed_s <= 0:
        return None
    return (total - done) * elapsed_s / done


def fmt_bytes(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


def fmt_s(s):
    if s is None:
        return "--:--"
    s = int(round(s))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def bar(done, total, width=30):
    frac = min(1.0, done / total) if total else 0.0
    full = int(frac * width)
    return "[" + "#" * full + "-" * (width - full) + f"] {100 * frac:5.1f}%"


# ---------------------------------------------------------------- host rotation (--auto)
# When a host counts as bad (seconds). Module constants so tests can shrink them.
LOADING_LIMIT_S = 6 * 60      # still "loading" this long after renting
SSH_LIMIT_S = 3 * 60          # ssh not up this long after "running"
UPLOAD_WINDOW_S = 30          # upload speed is judged on its first 30 s
STALL_LIMIT_S = 3 * 60        # upscale progress unchanged this long
MIN_RENT_H = 0.2              # rent a host only if the budget left covers at least this long at its price


def upload_too_slow(bytes_sent, elapsed_s, min_mbps, window_s=None):
    """True once the first window_s seconds of an upload averaged under min_mbps megabits per second.
    None until the window is over (or if the upload finished before it)."""
    window_s = UPLOAD_WINDOW_S if window_s is None else window_s
    if elapsed_s < window_s:
        return None
    return bytes_sent * 8 / elapsed_s / 1e6 < min_mbps


def geo_skipped(geolocation, patterns):
    g = (geolocation or "").lower()
    return any(p.lower() in g for p in patterns if p)


def pick_offer(offers, skip_machines, skip_geos):
    """The first offer not on a skipped machine or in a skipped location (offers are cheapest first)."""
    skip = {str(m) for m in skip_machines}
    for o in offers:
        if str(o.get("machine_id")) not in skip and not geo_skipped(o.get("geolocation"), skip_geos):
            return o
    return None


def parse_deadline(hhmm, now):
    """Epoch seconds of the next HH:MM local time at or after now (so 23:30 at 20:00 is tonight, at 23:45 tomorrow)."""
    if not hhmm:
        return None
    h, m = (int(x) for x in hhmm.split(":"))
    lt = time.localtime(now)
    t = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))
    return t if t >= now else t + 24 * 3600


class Budget:
    """Spend and limits across every host of one job: --max-usd covers all attempts together."""

    def __init__(self, max_usd, max_hours, max_attempts, deadline=None, t0=None, margin_usd=0.15):
        self.max_usd, self.max_hours, self.max_attempts, self.deadline = max_usd, max_hours, max_attempts, deadline
        self.t0 = time.time() if t0 is None else t0
        self.margin = margin_usd  # billing runs a few seconds past our clock (destroy); stay this far under the cap
        self.attempts = []        # {"machine_id", "cost_usd", "reason"}

    @property
    def spent_before(self):
        return sum(a["cost_usd"] for a in self.attempts)

    def record(self, machine_id, cost_usd, reason=None):
        self.attempts.append({"machine_id": machine_id, "cost_usd": cost_usd, "reason": reason})

    def cap(self):
        return self.max_usd - self.margin

    def why_not_rent(self, dph, now, min_hours=None):
        """None if another host may be rented at dph $/h now, else the reason not to."""
        min_hours = MIN_RENT_H if min_hours is None else min_hours
        if len(self.attempts) >= self.max_attempts:
            return f"{self.max_attempts} attempts used"
        if self.deadline is not None and now >= self.deadline:
            return "past the deadline"
        if now - self.t0 >= self.max_hours * 3600:
            return f"--max-hours {self.max_hours} reached"
        if self.cap() - self.spent_before < dph * min_hours:
            return f"${self.cap() - self.spent_before:.2f} left is under {min_hours * 60:.0f} min at ${dph:.2f}/h"
        return None

    def must_stop(self, spent_now, now, reserve_usd=0.0, reserve_s=0.0):
        """Why the current host must stop upscaling and download (cap, time or deadline, counting the download still
        to do), or None."""
        if self.spent_before + spent_now + reserve_usd >= self.cap():
            return "budget"
        if now + reserve_s - self.t0 >= self.max_hours * 3600:
            return "max-hours"
        if self.deadline is not None and now + reserve_s >= self.deadline:
            return "deadline"
        return None


def load_skips(path):
    """{machine id: {"reason", "time", ...}} from the persistent skip list (missing file: empty)."""
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def add_skip(path, machine_id, reason, **info):
    skips = load_skips(path)
    skips[str(machine_id)] = {"reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    json.dump(skips, open(tmp, "w"), indent=1); os.replace(tmp, path)
    return skips


def status_line(s):
    """One plain-text status.log line from a status dict."""
    prog = ""
    if s.get("total"):
        prog = (f"{fmt_bytes(s['done'])}/{fmt_bytes(s['total'])}" if s.get("unit") == "bytes"
                else f"{s['done']}/{s['total']} pages")
    parts = [s["time"], f"attempt {s['attempt'] or '-'}/{s['attempts']}",
             f"machine {s.get('machine_id') or '-'} ({s.get('location') or '-'}, {s.get('gpu') or '-'}, ${s.get('dph') or 0:.2f}/h)",
             s.get("phase") or "-", prog or "-",
             f"{s['speed_mb_s']:.1f} MB/s" if s.get("speed_mb_s") is not None else "-",
             f"ETA {fmt_s(s.get('eta_s'))}", f"spent ${s.get('spent_usd', 0):.2f}"]
    if s.get("last_drop"):
        parts.append(f"last dropped {s['last_drop']['machine_id']}: {s['last_drop']['reason']}")
    return " | ".join(parts)


class StatusLog:
    """status.log (one line per `every` seconds, and on every event) and status.json (latest state), flushed each write."""

    def __init__(self, out, snapshot, every=60, clock=time.time):
        self.log_path, self.json_path = os.path.join(out, "status.log"), os.path.join(out, "status.json")
        self.snapshot, self.every, self.clock, self.last = snapshot, every, clock, None
        os.makedirs(out, exist_ok=True)

    def write(self, final=None):
        now = self.clock()
        s = self.snapshot()
        s["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        if final:
            s["final"] = final
        line = status_line(s) if not final else f"{s['time']} | {final}"
        with open(self.log_path, "a") as f:
            f.write(line + "\n"); f.flush(); os.fsync(f.fileno())
        tmp = self.json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(s, f, indent=1); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.json_path)
        self.last = now

    def tick(self):
        """Writes if `every` seconds have passed since the last line."""
        if self.last is None or self.clock() - self.last >= self.every:
            self.write()
