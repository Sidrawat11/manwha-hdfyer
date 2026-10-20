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


def vast(*args, check=True, timeout=120):
    out = sh(["vastai", *args, "--raw"], check=check, timeout=timeout)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def credit():
    user = vast("show", "user")
    return user.get("credit", user.get("balance")) if isinstance(user, dict) else None


def search(query, skip_machines=()):
    """Offers matching a Vast query, cheapest first, without the machines in skip_machines (hosts that failed before)."""
    offers = vast("search", "offers", query, "-o", "dph_total", check=False)
    return drop_machines(offers if isinstance(offers, list) else [], skip_machines)


def drop_machines(offers, skip_machines):
    skip = {str(m) for m in skip_machines}
    return [o for o in offers if str(o.get("machine_id")) not in skip]


def rent(offer_id, image, disk_gb, label):
    res = vast("create", "instance", str(offer_id), "--image", image, "--disk", str(disk_gb), "--ssh", "--direct", "--label", label)
    iid = res.get("new_contract") if isinstance(res, dict) else None
    if not iid:
        raise RuntimeError(f"create failed: {res}")
    return iid


RUNNING_POLL_S = 10


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
        time.sleep(RUNNING_POLL_S)


def ssh_args(userhost, port):
    """Plain ssh to the instance. ConnectTimeout: a dead host fails in 10 s; BatchMode: never prompt for a password or
    passphrase (a prompt would wait forever); ServerAlive: a connection that dies mid-transfer ends within ~90 s."""
    return ["ssh", "-p", str(port), "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", userhost]


def ssh_command(iid):
    """Plain ssh argv for the instance (see ssh_args)."""
    userhost, port = sh(["vastai", "ssh-url", str(iid)], timeout=120).split("://", 1)[1].rsplit(":", 1)
    return ssh_args(userhost, port)


def with_control(ssh, control_path):
    """ssh multiplexed over one master connection (fast repeated polls). Only after wait_ssh has succeeded once:
    a master that hangs while connecting would make every later call hang too."""
    return ssh[:-1] + ["-o", "ControlMaster=auto", "-o", f"ControlPath={control_path}", "-o", "ControlPersist=600", ssh[-1]]


def on_host(ssh, cmd, stdin=False):
    """The argv that runs cmd on the instance; without stdin, ssh gets -n (it must come before the host)."""
    return ssh[:-1] + ([] if stdin else ["-n"]) + [ssh[-1], cmd]


def ssh_failure(returncode, stderr):
    """The reason a probe failed, for the live view: ssh's last stderr line, e.g. "Permission denied (publickey)."."""
    lines = [l.strip() for l in (stderr or "").splitlines() if l.strip()]
    return lines[-1] if lines else f"ssh exited with {returncode}"


def wait_ssh(ssh, tries=40, attempt_timeout=30, pause_s=10, on_fail=None, max_s=None):
    """Retries `ssh true` until it works. Each attempt is killed after attempt_timeout seconds, so one hung attempt
    cannot block the wait; on_fail(attempt, reason) is told why each attempt failed. With max_s, gives up once that
    many seconds have passed (each attempt's timeout is cut to fit)."""
    reason, t0 = "no attempt made", time.time()
    for k in range(1, tries + 1):
        if max_s is not None:
            left = max_s - (time.time() - t0)
            if left <= 0:
                raise RuntimeError(f"ssh not up after {max_s / 60:.0f} min; last: {reason}")
            attempt_timeout = min(attempt_timeout, max(1, left))
        try:
            r = subprocess.run(on_host(ssh, "true"), capture_output=True, text=True, timeout=attempt_timeout)
            if r.returncode == 0:
                return
            reason = ssh_failure(r.returncode, r.stderr)
        except subprocess.TimeoutExpired:
            reason = f"no answer within {attempt_timeout} s"
        if on_fail:
            on_fail(k, reason)
        if k < tries:
            time.sleep(pause_s if max_s is None else max(0, min(pause_s, max_s - (time.time() - t0))))
    raise RuntimeError(f"ssh never came up after {tries} attempts; last: {reason}")


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


def detached_command(cmd, cwd=None):
    """The remote line that starts cmd fully detached. Nothing but the setsid'd shell sits behind the `&`: a trailing &
    after an `a && b` chain backgrounds the whole chain as a subshell that keeps ssh's stdout open until cmd exits."""
    inner = f"cd {shlex.quote(cwd)} && {cmd}" if cwd else cmd
    return f"setsid nohup bash -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &"


def pgrep_command(pattern):
    """pgrep -f for pattern that cannot match the remote shell running it (first character in brackets)."""
    return "pgrep -f " + shlex.quote(f"[{pattern[0]}]{pattern[1:]}")


def start_detached(ssh, cmd, match, cwd=None, before=None, log=None, wait_s=10):
    """Runs `before` (if any) in its own ssh call and waits for it, starts cmd detached, then checks within wait_s
    seconds that a process matching `match` is running. Raises with the tail of `log` if it is not."""
    if before:
        r = subprocess.run(on_host(ssh, before), capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"before starting {match}: `{before}` failed:\n{r.stdout}{r.stderr}")
    subprocess.run(on_host(ssh, detached_command(cmd, cwd)), capture_output=True, timeout=30, check=True)
    deadline = time.time() + wait_s
    while True:
        if subprocess.run(on_host(ssh, pgrep_command(match)), capture_output=True, timeout=30).returncode == 0:
            return
        if time.time() > deadline:
            tail = sh(on_host(ssh, f"tail -n 30 {shlex.quote(log)}"), check=False, timeout=60) if log else ""
            raise RuntimeError(f"{match} is not running {wait_s} s after launch" + (f"; {log}:\n{tail}" if log else ""))
        time.sleep(1)


def destroy(iid, tries=5):
    """Destroys (not stops) the instance: that is what ends all billing. Returns True on success."""
    for _ in range(tries):
        try:
            if subprocess.run(["vastai", "destroy", "instance", str(iid), "-y"], capture_output=True, timeout=60).returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    return False


def write_cloud_info(local_out, iid, info, dph, t_start, t_end, **extra):
    os.makedirs(local_out, exist_ok=True)
    json.dump({**extra, "instance": iid, "gpu": info.get("gpu_name"), "geolocation": info.get("geolocation"),
               "dph": dph, "minutes_watched": round((t_end - t_start) / 60, 1),
               "cost_usd_approx": round((t_end - t_start) / 3600 * dph, 2),
               "finished_local": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(local_out, "cloud.json"), "w"), indent=1)
