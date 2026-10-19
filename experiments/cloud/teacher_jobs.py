"""Pure parts of the teacher upscale job: which pages, where their outputs go, what is left to do, how big it gets.

No GPU, no network; tests in tests/test_cloud.py.
"""
import math, os

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
