"""cloud: the pure parts of the teacher upscale job (experiments/cloud/teacher_jobs.py, upscale_teacher.py).

Path layout, what is left to do or fetch after an interruption, the size estimate and the caps. No network.
"""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

CLOUD = Path(__file__).resolve().parent.parent / "experiments" / "cloud"


def checks() -> list[tuple[str, bool]]:
    sys.path.insert(0, str(CLOUD))
    sys.path.insert(0, str(CLOUD.parent))
    import teacher_jobs as tj
    import upscale_teacher as ut
    checks = []

    train = {"train": {"train_L1.cbz": [{"entry": "000.jpg", "series_label": "L1", "cbz": "S1/Chapter 3_ab12.cbz", "page": "007__002.jpg"},
                                        {"entry": "001.jpg", "series_label": "L1", "cbz": "S1/Chapter 3_ab12.cbz", "page": "008.jpg"}],
                       "train_L2.cbz": [{"entry": "000.png", "series_label": "L2", "cbz": "S2/Ep 1.cbz", "page": "001.png"}]},
             "test": [{"entry": "000.jpg", "series_label": "L1", "cbz": "S1/Chapter 9_cd34.cbz", "page": "012.jpg"}]}
    exams = {"B": {"test": [{"entry": "000.png", "cbz": "X/c.cbz", "page": "014.png"}]},
             "A": {"test": [{"entry": "000.png", "cbz": "Y/c.cbz", "page": "003.png"},
                            {"entry": "001.webp", "cbz": "Y/c.cbz", "page": "004.png"}]}}
    jobs = tj.jobs_from_manifests(train, exams)
    outs = [j["out"] for j in jobs]
    checks.append(("layout: training pages go to training-v3/<label>/<chapter>/<page>.png, test pages too, exams to exams/<label>/<entry>.png",
                   outs == ["training-v3/L1/Chapter 3_ab12/007__002.png", "training-v3/L1/Chapter 3_ab12/008.png",
                            "training-v3/L2/Ep 1/001.png", "training-v3/L1/Chapter 9_cd34/012.png",
                            "exams/A/000.png", "exams/A/001.png", "exams/B/000.png"]))
    checks.append(("layout: each job reads its page from the right CBZ entry",
                   [(j["src"], j["entry"]) for j in jobs] == [
                       ("training/train_L1.cbz", "000.jpg"), ("training/train_L1.cbz", "001.jpg"), ("training/train_L2.cbz", "000.png"),
                       ("training/test.cbz", "000.jpg"), ("exams/A/test.cbz", "000.png"), ("exams/A/test.cbz", "001.webp"),
                       ("exams/B/test.cbz", "000.png")]))
    clash = json.loads(json.dumps(train))
    clash["test"][0].update(cbz="S1/Chapter 3_ab12.cbz", page="008.png")
    try:
        tj.jobs_from_manifests(clash, {}); refused = False
    except ValueError:
        refused = True
    checks.append(("layout: two pages mapping to one output file are refused", refused))
    checks.append(("layout: page counts per set and label",
                   tj.page_counts(jobs) == {"training-v3": {"L1": 3, "L2": 1}, "exams": {"A": 2, "B": 1}}))

    have = {outs[0]: 900, outs[1]: 0, outs[4]: 12}
    checks.append(("resume: pages already downloaded are not sent again; an empty file counts as missing",
                   [j["out"] for j in tj.pending(jobs, have)] == [outs[1], outs[2], outs[3], outs[5], outs[6]]))
    remote = {outs[0]: 900, outs[1]: 500, outs[4]: 40, outs[2]: 7}
    checks.append(("resume: fetch what is missing locally or differs in size (a copy cut off half-way), nothing else",
                   tj.to_download(remote, have) == sorted([outs[1], outs[2], outs[4]])))
    checks.append(("resume: nothing to fetch once local matches remote", tj.to_download(remote, dict(remote)) == []))

    many = [{"out": str(k)} for k in range(100)]
    pick = [j["out"] for j in tj.smoke_pick(many, 10)]
    checks.append(("smoke: 10 pages spread over the whole list, first and last included, no repeats",
                   len(set(pick)) == 10 and pick[0] == "0" and pick[-1] == "99" and pick == sorted(pick, key=int)))
    checks.append(("smoke: a short list is taken whole", tj.smoke_pick(many[:4], 10) == many[:4] and tj.smoke_pick(many, 1) == many[:1]))

    sample = [(1000, 3000), (3000, 5000)]  # 2 bytes per input pixel
    checks.append(("estimate: bytes per input pixel of the sample times the pages' pixels",
                   tj.estimate_bytes(sample, [500, 1500, 2000]) == 8000))
    checks.append(("disk: refused below 1.5x the estimate, allowed at it",
                   tj.disk_ok(1500, 1000) and not tj.disk_ok(1499, 1000)))
    checks.append(("disk: the instance gets inputs + 1.5x outputs + 20 GB, at least 40 GB",
                   tj.remote_disk_gb(1e9, 1e9) == 40 and tj.remote_disk_gb(2e9, 30e9) == 67))

    checks.append(("caps: spend is $/h times hours", abs(tj.spend(3.6, 1800) - 1.8) < 1e-9))
    checks.append(("caps: stop when spend plus the download still to pay reaches --max-usd",
                   not tj.over_budget(8.0, 600, 10, 3, reserve_usd=1.9) and tj.over_budget(8.0, 600, 10, 3, reserve_usd=2.0)))
    checks.append(("caps: stop when time plus the download still to do reaches --max-hours",
                   not tj.over_budget(1, 3 * 3600 - 301, 10, 3, reserve_s=300) and tj.over_budget(1, 3 * 3600 - 300, 10, 3, reserve_s=300)))
    checks.append(("progress: ETA from the rate so far; none before the first item",
                   tj.eta_s(25, 100, 50) == 150 and tj.eta_s(0, 100, 50) is None))
    checks.append(("progress: bar and times render", tj.bar(1, 4, 8) == "[##------]  25.0%" and tj.fmt_s(3725) == "1:02:05"
                   and tj.fmt_s(65) == "01:05" and tj.fmt_s(None) == "--:--" and tj.fmt_bytes(6.1e9) == "6.1 GB"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "train").mkdir(); (tmp / "exams" / "A").mkdir(parents=True)
        with zipfile.ZipFile(tmp / "train" / "train_L1.cbz", "w") as z:
            for n in ("000.jpg", "001.jpg", "002.jpg"):
                z.writestr(n, n.encode() * 50)
        with zipfile.ZipFile(tmp / "exams" / "A" / "test.cbz", "w") as z:
            z.writestr("000.png", b"exam")
        old = ut.TRAIN, ut.EXAMS
        ut.TRAIN, ut.EXAMS = str(tmp / "train"), str(tmp / "exams")
        try:
            todo = [{"src": "training/train_L1.cbz", "entry": "001.jpg"}, {"src": "exams/A/test.cbz", "entry": "000.png"}]
            files = ut.upload_inputs(todo, str(tmp))
            part = zipfile.ZipFile(files["in/training/train_L1.cbz"])
            checks.append(("upload: a CBZ only partly needed goes up as just those entries, byte for byte; a whole one as is",
                           part.namelist() == ["001.jpg"] and part.read("001.jpg") == b"001.jpg" * 50
                           and files["in/exams/A/test.cbz"] == str(tmp / "exams" / "A" / "test.cbz")))
            out = tmp / "out"
            (out / "exams" / "A").mkdir(parents=True)
            (out / "exams" / "A" / "000.png").write_bytes(b"12345")
            (out / "exams" / "A" / "001.png.tmp.png").write_bytes(b"12")
            (out / "exams" / "A" / "002.png.part").write_bytes(b"12")
            checks.append(("resume: local scan sees finished PNGs only, not temp or partial files",
                           ut.local_sizes(str(out)) == {"exams/A/000.png": 5}))
        finally:
            ut.TRAIN, ut.EXAMS = old
    return checks
