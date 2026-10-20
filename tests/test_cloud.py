"""cloud: the pure parts of the teacher upscale job (experiments/cloud/teacher_jobs.py, upscale_teacher.py).

Path layout, what is left to do or fetch after an interruption, the size estimate and the caps. No network.
"""

import json
import os
import subprocess
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

    import vast
    script = vast.probe_script()
    order = [script.find(c) for c in ("bash -lc python", "bash -lc python3", "/opt/conda/bin/python", "/opt/venv/bin/python",
                                      "/usr/local/bin/python", "which -a python3")]
    checks.append(("probe: tries a login shell first, then /opt/conda, /opt/venv, /usr/local, then every python3 on PATH",
                   all(k >= 0 for k in order) and order == sorted(order)))
    checks.append(("probe: one bash -c script that imports torch and checks CUDA, failing when nothing qualifies",
                   script.startswith("bash -c '") and "torch.cuda.is_available()" in script and script.endswith("exit 1'")))
    checks.append(("probe: the parse takes the PYPROBE line of an interpreter with CUDA",
                   vast.parse_probe("PYTRIED /usr/bin/python3: ModuleNotFoundError\nPYPROBE /opt/conda/bin/python 2.10.0+cu128 True\n")
                   == ("/opt/conda/bin/python", "2.10.0+cu128")))
    checks.append(("probe: torch without CUDA, a relative path or no output is not an answer",
                   vast.parse_probe("PYPROBE /usr/bin/python3 2.10.0 False") is None and vast.parse_probe("PYPROBE python 2.10.0 True") is None
                   and vast.parse_probe("") is None))
    checks.append(("probe: the command goes after the host with -n before it",
                   vast.on_host(["ssh", "-p", "22", "root@h"], "x") == ["ssh", "-p", "22", "-n", "root@h", "x"]))

    pip = vast.pip_command("/usr/bin/python", vast.BASE_PACKAGES)
    checks.append(("install: pip runs with the probed interpreter, --no-deps and --break-system-packages (PEP 668 system Python)",
                   pip.startswith("/usr/bin/python -m pip install -q --no-deps --break-system-packages ")))
    checks.append(("install: spandrel 0.4.2 and the pieces it needs beyond the image; never torch, torchvision or numpy",
                   pip.split("--break-system-packages ")[1].split() == ["spandrel==0.4.2", "einops", "safetensors", "typing_extensions",
                                                                       "opencv-python-headless", "tqdm"]
                   and not any(p.split("=")[0] in ("torch", "torchvision", "numpy") for p in vast.BASE_PACKAGES)))
    checks.append(("install: an interpreter path with a space is quoted", vast.pip_command("/opt/my py/python", ["tqdm"]).startswith("'/opt/my py/python' -m pip")))
    after = vast.parse_check("Warning: x\nPYCHECK /usr/bin/python 2.10.0+cu128 True 0.25.0+cu128\n")
    checks.append(("install: the re-check reads python, torch, CUDA and torchvision",
                   after == {"python": "/usr/bin/python", "torch": "2.10.0+cu128", "cuda": True, "torchvision": "0.25.0+cu128"}))
    checks.append(("install: same torch with CUDA passes", vast.install_problem("/usr/bin/python", "2.10.0+cu128", after) is None))
    checks.append(("install: a changed torch version fails",
                   "changed" in (vast.install_problem("/usr/bin/python", "2.9.1+cu128", after) or "")))
    checks.append(("install: CUDA lost fails", "CUDA" in (vast.install_problem("/usr/bin/python", "2.10.0+cu128", {**after, "cuda": False}) or "")))
    checks.append(("install: torchvision (or torch) not importing fails",
                   vast.parse_check("ModuleNotFoundError: No module named 'torchvision'") is None
                   and vast.install_problem("/usr/bin/python", "2.10.0+cu128", None) is not None))

    with tempfile.TemporaryDirectory() as tmp:  # run the script for real against fake interpreters
        tmp = Path(tmp)
        fakes = {"notorch": "echo \"ModuleNotFoundError: No module named 'torch'\" >&2; exit 1",
                 "nocuda": 'echo "PYPROBE $0 2.10.0 False"', "good": 'echo "PYPROBE $0 2.10.0+cu128 True"'}
        for name, body in fakes.items():
            (tmp / name).write_text("#!/bin/sh\n" + body + "\n"); (tmp / name).chmod(0o755)
        old = vast.PY_LOGIN, vast.PY_PATHS
        env = {**os.environ, "PATH": "/usr/bin:/bin"}
        try:
            vast.PY_LOGIN = ()
            vast.PY_PATHS = (str(tmp / "missing"), str(tmp / "notorch"), str(tmp / "nocuda"), str(tmp / "good"))
            r = subprocess.run(["sh", "-c", vast.probe_script()], capture_output=True, text=True, env=env)
            tried = [l.split(":")[0] for l in r.stdout.splitlines() if l.startswith("PYTRIED")]
            checks.append(("probe: skips a missing interpreter, one without torch and one without CUDA, picks the first good one",
                           r.returncode == 0 and vast.parse_probe(r.stdout) == (str(tmp / "good"), "2.10.0+cu128")
                           and tried == [f"PYTRIED {tmp / n}" for n in ("missing", "notorch", "nocuda")]))
            vast.PY_PATHS = (str(tmp / "notorch"),)
            r = subprocess.run(["sh", "-c", vast.probe_script()], capture_output=True, text=True, env=env)
            checks.append(("probe: with no good interpreter it exits non-zero and says why each failed",
                           r.returncode == 1 and vast.parse_probe(r.stdout) is None and f"PYTRIED {tmp / 'notorch'}: ModuleNotFoundError" in r.stdout))
        finally:
            vast.PY_LOGIN, vast.PY_PATHS = old

        fake_ssh = tmp / "ssh"
        fake_ssh.write_text('#!/bin/sh\nfor a; do last="$a"; done\nexec sh -c "$last"\n'); fake_ssh.chmod(0o755)
        for name, torch_after in (("steady", "2.10.0+cu128"), ("swapped", "2.11.0+cu130")):
            (tmp / name).write_text(f'#!/bin/sh\n[ "$1" = "-m" ] && {{ echo "$@" > {tmp}/{name}.pip; exit 0; }}\n'
                                    f'echo "PYCHECK $0 {torch_after} True 0.25.0"\n'); (tmp / name).chmod(0o755)
        got = vast.install_packages([str(fake_ssh), "host"], str(tmp / "steady"), "2.10.0+cu128", log=str(tmp / "pip.log"))
        pip_args = (tmp / "steady.pip").read_text().split()
        checks.append(("install: runs pip on the instance with --no-deps, then reports torch and torchvision",
                       got == {"python": str(tmp / "steady"), "torch": "2.10.0+cu128", "cuda": True, "torchvision": "0.25.0"}
                       and pip_args[:5] == ["-m", "pip", "install", "-q", "--no-deps"] and "spandrel==0.4.2" in pip_args))
        try:
            vast.install_packages([str(fake_ssh), "host"], str(tmp / "swapped"), "2.10.0+cu128", log=str(tmp / "pip.log")); msg = ""
        except RuntimeError as e:
            msg = str(e)
        checks.append(("install: fails when pip changed torch", "torch changed from 2.10.0+cu128 to 2.11.0+cu130" in msg))

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
