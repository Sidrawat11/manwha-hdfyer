"""Shared Vast.ai helpers for the cloud jobs (exp16 training, exp17 benchmark, the teacher upscale job).

Renting, waiting for the instance, SSH, and always destroying it. Needs the vastai CLI set up once:
  pip install vastai && vastai set api-key <key>  (and an SSH key registered with `vastai create ssh-key`)
"""
import json, os, shlex, subprocess, time

DEFAULT_IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"


def sh(cmd, check=True, capture=True, timeout=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip() if capture else ""


def vast(*args, check=True):
    out = sh(["vastai", *args, "--raw"], check=check)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def credit():
    user = vast("show", "user")
    return user.get("credit", user.get("balance")) if isinstance(user, dict) else None


def search(query):
    """Offers matching a Vast query, cheapest first."""
    offers = vast("search", "offers", query, "-o", "dph_total", check=False)
    return offers if isinstance(offers, list) else []


def rent(offer_id, image, disk_gb, label):
    res = vast("create", "instance", str(offer_id), "--image", image, "--disk", str(disk_gb), "--ssh", "--direct", "--label", label)
    iid = res.get("new_contract") if isinstance(res, dict) else None
    if not iid:
        raise RuntimeError(f"create failed: {res}")
    return iid


def wait_running(iid, timeout_s=900, on_status=None):
    start = time.time()
    while True:
        info = vast("show", "instance", str(iid), check=False)
        status = info.get("actual_status") if isinstance(info, dict) else None
        if on_status:
            on_status(status)
        if status == "running":
            return
        if time.time() - start > timeout_s:
            raise RuntimeError(f"instance not running after {timeout_s // 60} min")
        time.sleep(10)


def ssh_command(iid, control_path=None):
    """The ssh argument list for the instance. With control_path, connections are multiplexed (fast repeated polls)."""
    userhost, port = sh(["vastai", "ssh-url", str(iid)]).split("://", 1)[1].rsplit(":", 1)
    cmd = ["ssh", "-p", port, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR"]
    if control_path:
        cmd += ["-o", "ControlMaster=auto", "-o", f"ControlPath={control_path}", "-o", "ControlPersist=600"]
    return cmd + [userhost]


def on_host(ssh, cmd, stdin=False):
    """The argv that runs cmd on the instance; without stdin, ssh gets -n (it must come before the host)."""
    return ssh[:-1] + ([] if stdin else ["-n"]) + [ssh[-1], cmd]


def wait_ssh(ssh, tries=40):
    for _ in range(tries):
        if subprocess.run(on_host(ssh, "true"), capture_output=True).returncode == 0:
            return
        time.sleep(10)
    raise RuntimeError("ssh never came up")


# Where the image's PyTorch might live. Over ssh, a bare `python` can be the OS's own Python (PEP 668, no torch),
# not the image's interpreter, so the remote side always uses the absolute path find_python returns.
PY_LOGIN = ("python", "python3")  # tried in a login shell first (it sets up PATH as the image intends)
PY_PATHS = ("/opt/conda/bin/python", "/opt/venv/bin/python", "/usr/local/bin/python")
PROBE_PY = "import torch, sys; print('PYPROBE', sys.executable, torch.__version__, torch.cuda.is_available())"


def probe_script():
    """One remote shell script: tries each candidate in order and prints the first `PYPROBE <path> <torch> True` line;
    for each failure a `PYTRIED <candidate>: <last line of its output>` line, for the error message."""
    code = shlex.quote(PROBE_PY)
    tries = [(f"bash -lc {n}", f"bash -lc {shlex.quote(f'{n} -c {code}')}") for n in PY_LOGIN]
    tries += [(p, f"{p} -c {code}") for p in PY_PATHS]
    lines = [f'out=$({cmd} 2>&1); case "$out" in *"PYPROBE "*" True"*) echo "$out" | grep "^PYPROBE " | tail -n 1; exit 0;; esac; '
             f'echo "PYTRIED {label}: $(echo "$out" | tail -n 1)"' for label, cmd in tries]
    lines.append(f'for p in $(which -a python3 2>/dev/null); do out=$("$p" -c {code} 2>&1); case "$out" in *"PYPROBE "*" True"*) '
                 f'echo "$out" | grep "^PYPROBE " | tail -n 1; exit 0;; esac; echo "PYTRIED $p: $(echo "$out" | tail -n 1)"; done')
    lines.append("exit 1")
    return "bash -c " + shlex.quote("\n".join(lines))


def parse_probe(output):
    """(interpreter path, torch version) from the probe's output, or None if no interpreter had torch with CUDA."""
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "PYPROBE" and parts[3] == "True" and parts[1].startswith("/"):
            return parts[1], parts[2]
    return None


def find_python(ssh):
    """The absolute path of the remote interpreter that has torch with CUDA, and its torch version. Raises if none."""
    r = subprocess.run(on_host(ssh, probe_script()), capture_output=True, text=True, timeout=180)
    found = parse_probe(r.stdout)
    if not found:
        tried = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]
        raise RuntimeError("no Python on the instance has torch with CUDA. Tried:\n  " + "\n  ".join(tried or ["(no output)"]))
    return found


# What the jobs add to the image. --no-deps, so pip never touches the image's torch, torchvision or numpy; every
# dependency a package needs beyond those is therefore named here. --break-system-packages: in the 2.10 image torch
# lives in the system Python, which is PEP 668 "externally managed"; fine in a throwaway container.
BASE_PACKAGES = ("spandrel==0.4.2", "einops", "safetensors", "typing_extensions", "opencv-python-headless", "tqdm")
CHECK_PY = ("import sys, torch, torchvision; "
            "print('PYCHECK', sys.executable, torch.__version__, torch.cuda.is_available(), torchvision.__version__)")


def pip_command(py, packages):
    return f"{shlex.quote(py)} -m pip install -q --no-deps --break-system-packages " + " ".join(shlex.quote(p) for p in packages)


def parse_check(output):
    """{"python", "torch", "cuda", "torchvision"} from the CHECK_PY line, or None."""
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == "PYCHECK":
            return {"python": parts[1], "torch": parts[2], "cuda": parts[3] == "True", "torchvision": parts[4]}
    return None


def install_problem(py, torch_before, check):
    """Why the interpreter is no longer usable after the install, or None if torch and CUDA are as probed."""
    if check is None:
        return "torch or torchvision no longer imports"
    if check["torch"] != torch_before:
        return f"torch changed from {torch_before} to {check['torch']}"
    if not check["cuda"]:
        return "CUDA is no longer available to torch"
    return None


def install_packages(ssh, py, torch_ver, packages=BASE_PACKAGES, log="/workspace/m/pip.log", timeout=1800):
    """pip-installs the packages into py, then re-checks it: same torch, CUDA still available, torchvision imports.
    Returns {"python", "torch", "cuda", "torchvision"} for the manifest; raises with the reason otherwise."""
    cmd = (f"mkdir -p {shlex.quote(os.path.dirname(log))} && {pip_command(py, packages)} > {shlex.quote(log)} 2>&1 "
           f"|| {{ tail -n 30 {shlex.quote(log)}; exit 1; }}")
    r = subprocess.run(on_host(ssh, cmd), capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"pip install failed on the instance:\n{r.stdout}{r.stderr}")
    r = subprocess.run(on_host(ssh, f"{shlex.quote(py)} -c {shlex.quote(CHECK_PY)} 2>&1"), capture_output=True, text=True, timeout=180)
    check = parse_check(r.stdout)
    problem = install_problem(py, torch_ver, check)
    if problem:
        raise RuntimeError(f"after pip install, {py}: {problem}\n{r.stdout.strip()[-2000:]}")
    return check


def destroy(iid, tries=5):
    """Destroys (not stops) the instance: that is what ends all billing. Returns True on success."""
    for _ in range(tries):
        if subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True).returncode == 0:
            return True
        time.sleep(5)
    return False


def write_cloud_info(local_out, iid, info, dph, t_start, t_end, **extra):
    os.makedirs(local_out, exist_ok=True)
    json.dump({**extra, "instance": iid, "gpu": info.get("gpu_name"), "geolocation": info.get("geolocation"),
               "dph": dph, "minutes_watched": round((t_end - t_start) / 60, 1),
               "cost_usd_approx": round((t_end - t_start) / 3600 * dph, 2),
               "finished_local": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(local_out, "cloud.json"), "w"), indent=1)
