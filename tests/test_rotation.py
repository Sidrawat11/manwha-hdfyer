"""rotation: upscale_teacher --auto against a stand-in vastai and ssh (no network, no GPU, no money).

Each stand-in machine behaves as the scenario says: "ok", "loading" (never starts), "slowup" (upload throttled to
~0.2 MB/s), "stall" (the worker stops reporting). The limits are shrunk to seconds, the worker is a fake that writes
dummy PNGs, and dph is high so the spend caps bite within seconds.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

CLOUD = Path(__file__).resolve().parent.parent / "experiments" / "cloud"

FAKE_VASTAI = r'''#!/usr/bin/env python3
import json, os, shutil, sys
d = os.environ["FAKE_DIR"]; sc = json.load(open(os.path.join(d, "scenario.json")))
a = [x for x in sys.argv[1:] if x != "--raw"]
open(os.path.join(d, "vastai.log"), "a").write(" ".join(a) + "\n")
state_p = os.path.join(d, "state.json")
state = json.load(open(state_p)) if os.path.exists(state_p) else {}
def behavior(iid):
    return sc["behavior"].get(str(state.get(str(iid))), "ok")
if a[:2] == ["search", "offers"]:
    print(json.dumps(sc["offers"]))
elif a[:2] == ["create", "instance"]:
    offer = next(o for o in sc["offers"] if str(o["id"]) == a[2])
    iid = 1000 + len(state); state[str(iid)] = offer["machine_id"]; json.dump(state, open(state_p, "w"))
    shutil.rmtree(os.environ["FAKE_REMOTE"], ignore_errors=True); os.makedirs(os.environ["FAKE_REMOTE"])
    print(json.dumps({"success": True, "new_contract": iid}))
elif a[:2] == ["show", "instance"]:
    print(json.dumps({"actual_status": "loading" if behavior(a[2]) == "loading" else "running"}))
elif a[0] == "ssh-url":
    print(f"ssh://root@m{state[a[1]]}:22")
elif a[:2] == ["destroy", "instance"]:
    print("destroyed")
else:
    print("{}")
'''

FAKE_SSH = r'''#!/bin/bash
for a; do prev="$last"; last="$a"; done
[[ " $* " == *" -O "* ]] && exit 0
machine="${prev#root@m}"
mode=$(python3 -c "import json,sys; print(json.load(open('$FAKE_DIR/scenario.json'))['behavior'].get(sys.argv[1], 'ok'))" "$machine")
export FAKE_MODE="$mode"
if [ "$mode" = slowup ] && [[ "$last" == *"tar xf -"* ]]; then
  python3 -c "
import sys, time
while True:
    b = sys.stdin.buffer.read(20000)
    if not b: break
    sys.stdout.buffer.write(b); sys.stdout.buffer.flush(); time.sleep(0.1)" | bash -c "$last"
  exit $?
fi
exec bash -c "$last"
'''

FAKE_WORKER = r'''import json, os, sys, time
jobs = json.load(open("jobs.json")); mode = os.environ.get("FAKE_MODE", "ok")
p = {"phase": "upscaling", "total": len(jobs), "done": 0, "skipped": 0, "batch": None}
def save():
    json.dump(p, open("progress.tmp", "w")); os.replace("progress.tmp", "progress.json")
save()
for k, j in enumerate(jobs):
    if mode == "stall" and k == 1:
        time.sleep(600)
    dst = os.path.join("out", j["out"]); os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(os.urandom(3000)); p["done"] += 1; save(); time.sleep(0.05)
p["phase"] = "finished"; save()
'''


def _data(tmp: Path):
    """A tiny training set (2 CBZs, 6 pages) and one exam (2 pages), manifests as rebuild_datasets writes them."""
    rng = np.random.default_rng(3)
    train, exams = tmp / "training", tmp / "exams"
    (exams / "A").mkdir(parents=True); train.mkdir()
    def cbz(path, names):
        with zipfile.ZipFile(path, "w") as z:
            for n in names:
                z.writestr(n, cv2.imencode(".png", rng.integers(0, 255, (40, 30, 3), dtype=np.uint8))[1].tobytes())
    cbz(train / "train_L1.cbz", ["000.png", "001.png", "002.png"]); cbz(train / "test.cbz", ["000.png"])
    cbz(train / "train_L2.cbz", ["000.png", "001.png"]); cbz(exams / "A" / "test.cbz", ["000.png", "001.png"])
    e = lambda entry, label, ch, page: {"entry": entry, "series_label": label, "cbz": f"S/{ch}.cbz", "page": page}
    json.dump({"train": {"train_L1.cbz": [e(f"00{k}.png", "L1", "c1", f"p{k}.png") for k in range(3)],
                         "train_L2.cbz": [e(f"00{k}.png", "L2", "c2", f"p{k}.png") for k in range(2)]},
               "test": [e("000.png", "L1", "c9", "p0.png")]}, open(train / "manifest.json", "w"))
    json.dump({"test": [{"entry": "000.png"}, {"entry": "001.png"}]}, open(exams / "A" / "manifest.json", "w"))
    return train, exams


class Harness:
    def __init__(self, tmp: Path):
        sys.path.insert(0, str(CLOUD)); sys.path.insert(0, str(CLOUD.parent))
        import teacher_jobs as tj, upscale_teacher as ut, vast
        self.tj, self.ut, self.vast, self.tmp = tj, ut, vast, tmp
        self.bin = tmp / "bin"; self.bin.mkdir()
        for name, body in (("vastai", FAKE_VASTAI), ("ssh", FAKE_SSH)):
            (self.bin / name).write_text(body); (self.bin / name).chmod(0o755)
        (tmp / "teacher_worker.py").write_text(FAKE_WORKER)
        (tmp / "model.pth").write_bytes(os.urandom(200_000))
        self.train, self.exams = _data(tmp)
        self.remote = tmp / "remote"
        self.saved = {}

    def patch(self, mod, **kw):
        for k, v in kw.items():
            self.saved.setdefault((mod, k), getattr(mod, k)); setattr(mod, k, v)

    def __enter__(self):
        tj, ut, vast = self.tj, self.ut, self.vast
        self.patch(tj, LOADING_LIMIT_S=2, SSH_LIMIT_S=3, UPLOAD_WINDOW_S=1, STALL_LIMIT_S=3, MIN_RENT_H=0.0001)
        self.patch(vast, RUNNING_POLL_S=0.3,
                   find_python=lambda ssh: (sys.executable, "2.10.0+cu128"),
                   install_packages=lambda ssh, py, tv, *a, **k: {"python": py, "torch": tv, "cuda": True, "torchvision": "0.25.0"})
        self.patch(ut, TRAIN=str(self.train), EXAMS=str(self.exams), MODEL=str(self.tmp / "model.pth"), REMOTE=str(self.remote),
                   WORKER=str(self.tmp / "teacher_worker.py"), POLL_S=0.2, STATUS_EVERY_S=1,
                   SKIP_FILE=str(self.tmp / "skips.json"),
                   sample_sizes=lambda sample, sha, cache: [(1200, 3000)])
        self.env = {k: os.environ.get(k) for k in ("PATH", "FAKE_DIR", "FAKE_REMOTE")}
        os.environ.update(PATH=f"{self.bin}:{os.environ['PATH']}", FAKE_DIR=str(self.tmp), FAKE_REMOTE=str(self.remote))
        return self

    def __exit__(self, *exc):
        for (mod, k), v in self.saved.items():
            setattr(mod, k, v)
        for k, v in self.env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        subprocess.run(["pkill", "-f", str(self.tmp / "remote")], capture_output=True)  # a stalled fake worker

    def run(self, offers, behavior, *args, out="out"):
        json.dump({"offers": offers, "behavior": {str(k): v for k, v in behavior.items()}}, open(self.tmp / "scenario.json", "w"))
        for f in ("vastai.log", "state.json"):
            (self.tmp / f).unlink(missing_ok=True)
        sys.argv = ["upscale_teacher.py", "--auto", "--out", str(self.tmp / out), *args]
        cwd, buf = os.getcwd(), io.StringIO()
        t = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                code = self.ut.main()
        finally:
            os.chdir(cwd)
        log = (self.tmp / "vastai.log").read_text() if (self.tmp / "vastai.log").exists() else ""
        status = (self.tmp / out / "status.log").read_text() if (self.tmp / out / "status.log").exists() else ""
        st = json.load(open(self.tmp / out / "status.json")) if (self.tmp / out / "status.json").exists() else {}
        return {"code": code, "out": buf.getvalue(), "creates": [l.split()[2] for l in log.splitlines() if l.startswith("create instance")],
                "destroys": log.count("destroy instance"), "status": status, "status_json": st, "secs": time.time() - t,
                "pngs": sum(1 for _ in (self.tmp / out).rglob("*.png"))}


def offer(oid, machine, geo="Spain, ES", dph=36.0):
    return {"id": oid, "machine_id": machine, "geolocation": geo, "gpu_name": "H200_NVL", "dph_total": dph}


def checks() -> list[tuple[str, bool]]:
    checks = []
    with tempfile.TemporaryDirectory() as tmp, Harness(Path(tmp)) as h:
        tj = h.tj
        r = h.run([offer(1, 11), offer(2, 12)], {11: "slowup", 12: "ok"})
        checks.append(("rotate: a slow upload drops the host and the next one finishes all 8 pages",
                       r["code"] == 0 and r["creates"] == ["1", "2"] and r["destroys"] == 2 and r["pngs"] == 8
                       and "upload" in json.load(open(h.tmp / "skips.json"))["11"]["reason"]))
        checks.append(("status: status.json says DONE, with the dropped host and why",
                       r["status_json"].get("final", "").startswith("DONE 8/8 pages")
                       and r["status_json"]["last_drop"]["machine_id"] == 11 and "Mbit/s" in r["status_json"]["last_drop"]["reason"]))
        lines = r["status"].strip().splitlines()
        checks.append(("status: status.log has a line per phase change and per drop, and a final DONE line",
                       len(lines) >= 10 and lines[-1].split(" | ")[1].startswith("DONE")
                       and any("last dropped 11" in l for l in lines) and any("attempt 2/5" in l for l in lines)))

        r = h.run([offer(3, 13), offer(4, 14)], {13: "loading", 14: "ok"}, out="out2")
        checks.append(("rotate: a host still loading after the limit is dropped and the next one used",
                       r["code"] == 0 and r["creates"] == ["3", "4"] and r["pngs"] == 8
                       and "still loading" in json.load(open(h.tmp / "skips.json"))["13"]["reason"]))

        r = h.run([offer(1, 11), offer(5, 15)], {15: "ok"}, out="out3")
        checks.append(("skip list: a machine dropped in an earlier run is not rented again (persisted with reason and time)",
                       r["creates"] == ["5"] and r["code"] == 0 and set(json.load(open(h.tmp / "skips.json"))) >= {"11", "13"}
                       and "time" in json.load(open(h.tmp / "skips.json"))["11"]))

        r = h.run([offer(6, 16, geo="New Mexico, US"), offer(7, 17, geo="Japan, JP")], {17: "ok"}, out="out4")
        checks.append(("--skip-geo: New Mexico is skipped by default", r["creates"] == ["7"]))

        r = h.run([offer(8, 18), offer(9, 19)], {18: "stall", 19: "ok"}, out="out5")
        checks.append(("rotate: a stalled upscale downloads the finished page, drops the host, the next host does the rest",
                       r["code"] == 0 and r["creates"] == ["8", "9"] and r["pngs"] == 8
                       and "stalled" in json.load(open(h.tmp / "skips.json"))["18"]["reason"]))

        # $3600/h = $1/s: each stuck host costs ~2-3 s; a $5 cap allows only what fits, across all hosts together
        r = h.run([offer(20 + k, 30 + k, dph=3600.0) for k in range(5)], {30 + k: "loading" for k in range(5)},
                  "--max-usd", "5", out="out6")
        spent = r["status_json"].get("spent_usd", 99)
        checks.append(("caps: --max-usd covers all hosts together; renting stops before the cap and never passes it by more than ~1 s of billing",
                       r["code"] == 1 and 1 <= len(r["creates"]) < 5 and spent <= 5 + 1.2 and r["destroys"] == len(r["creates"])
                       and "FAILED" in r["status_json"].get("final", "")))

        r = h.run([offer(40 + k, 50 + k) for k in range(3)], {50 + k: "loading" for k in range(3)}, "--max-attempts", "2", out="out7")
        checks.append(("caps: --max-attempts bounds the hosts tried", r["creates"] == ["40", "41"] and r["code"] == 1
                       and "2 attempts used" in r["status_json"].get("final", "")))

        orig = tj.parse_deadline
        tj.parse_deadline = lambda hhmm, now: now + 1.5
        try:
            r = h.run([offer(60 + k, 70 + k) for k in range(3)], {70 + k: "loading" for k in range(3)}, "--deadline", "23:59", out="out8")
        finally:
            tj.parse_deadline = orig
        checks.append(("deadline: no new host is rented after it", len(r["creates"]) == 1 and r["code"] == 1
                       and "deadline" in r["status_json"].get("final", "")))

    clock = [1000.0]
    with tempfile.TemporaryDirectory() as tmp:
        log = tj.StatusLog(tmp, lambda: {"attempt": 1, "attempts": 5, "phase": "Upscaling", "done": 3, "total": 9, "unit": "pages",
                                         "spent_usd": 1.5}, every=60, clock=lambda: clock[0])
        for t in (1000, 1030, 1060, 1061, 1125, 1179, 1185):
            clock[0] = t; log.tick()
        log.write()  # an event (phase change, drop) always writes
        n = len(open(os.path.join(tmp, "status.log")).read().splitlines())
        checks.append(("status: one status.log line per 60 s of ticks, plus one per event", n == 4 + 1))
    sys.path.insert(0, str(CLOUD)); import upscale_teacher as ut
    live = ut.Live(10, 3, 5)
    live.new_host({"dph_total": 3.6, "inet_up_cost": 0.0156, "inet_down_cost": 0.013}, 1)
    live.sent, live.received = 1e9, 6.5e9
    checks.append(("spend: per-GB traffic is counted, uploads at inet_down_cost and downloads at inet_up_cost",
                   abs(live.spent_current() - (0.013 + 6.5 * 0.0156)) < 1e-9))
    budget = tj.Budget(10, 3, 5, deadline=time.time() - 1)
    checks.append(("deadline: Budget refuses a new host after the deadline", budget.why_not_rent(4.0, time.time()) == "past the deadline"))
    lt = time.localtime(time.mktime((2026, 10, 19, 20, 25, 0, 0, 0, -1)))
    now = time.mktime(lt)
    checks.append(("deadline: 23:30 at 20:25 is tonight, 20:00 at 20:25 is tomorrow",
                   tj.parse_deadline("23:30", now) - now == 3 * 3600 + 5 * 60 and tj.parse_deadline("20:00", now) - now == 24 * 3600 - 25 * 60))
    checks.append(("upload: judged on its first 30 s, in megabits per second",
                   tj.upload_too_slow(5e6, 10, 50) is None and tj.upload_too_slow(150e6, 30, 50) is True
                   and tj.upload_too_slow(300e6, 30, 50) is False))
    return checks
