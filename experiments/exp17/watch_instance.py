"""Live view of a running Vast.ai job (Exp 16 training or Exp 17 bench). Read-only: never stops or destroys anything.

  python experiments/exp17/watch_instance.py            # the newest running instance
  python experiments/exp17/watch_instance.py 55873624   # a specific one
Ctrl+C quits the viewer; the instance and the script that owns it keep running.
"""
import json, subprocess, sys, time

REMOTE = "/workspace/m"
PROBE = f"""cd {REMOTE} 2>/dev/null || {{ echo 'phase: no files yet (still uploading?)'; exit; }}
P=$(ps -eo etime,args | grep -E 'pip install|exp17/bench.py|exp16/distill.py|process_all' | grep -v grep | head -3 | cut -c1-110)
if echo "$P" | grep -q 'pip install'; then echo 'phase: INSTALLING packages'
elif echo "$P" | grep -qE 'bench.py|distill.py'; then echo 'phase: RUNNING'
elif ls runs/*/DONE >/dev/null 2>&1; then echo 'phase: FINISHED (waiting for download + destroy)'
else echo 'phase: IDLE or FAILED -> pip.log tail:'; tail -n 15 pip.log 2>/dev/null; fi
echo "$P" | sed 's/^/  proc /'
echo "  site-packages: $(du -sh /opt/conda/lib/python3*/site-packages 2>/dev/null | cut -f1)"
echo "  GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader 2>/dev/null)"
test -f run.log && tr '\\r' '\\n' < run.log | grep -v '^\\s*$' | tail -n 6 | cut -c1-170 | sed 's/^/  log  /'
"""


def main():
    iid = sys.argv[1] if len(sys.argv) > 1 else None
    if not iid:
        inst = json.loads(subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True).stdout or "[]")
        running = [i for i in inst if i.get("actual_status") == "running"]
        if not running:
            sys.exit("No running instances. (Nothing is costing money.)")
        iid = str(max(running, key=lambda i: i.get("start_date") or 0)["id"])
    url = subprocess.run(["vastai", "ssh-url", iid], capture_output=True, text=True).stdout.strip()
    userhost, port = url.split("://", 1)[1].rsplit(":", 1)
    ssh = ["ssh", "-n", "-p", port, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15", userhost]
    t0 = time.time()
    print(f"Watching instance {iid} ({userhost}:{port}). Ctrl+C to stop watching.\n", flush=True)
    while True:
        try:
            out = subprocess.run(ssh + [PROBE], capture_output=True, text=True, timeout=40).stdout.rstrip()
        except subprocess.TimeoutExpired:
            out = "(ssh timed out; retrying)"
        print(f"--- {time.strftime('%H:%M:%S')}  (+{(time.time() - t0) / 60:.1f} min watching)\n{out}\n", flush=True)
        if "No such" in out or not out:
            print("(instance unreachable: probably destroyed)", flush=True)
            break
        time.sleep(15)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped watching. The instance is untouched.")
