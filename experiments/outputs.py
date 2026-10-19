"""Generated results live in outputs/<expNN-short-name>/ (gitignored), each with a manifest.json saying what made them."""

import json, os, subprocess, sys, time

OUTPUTS = "outputs"


def output_dir(name: str, **info) -> str:
    """
    Creates outputs/<name>/ (or `name` itself if it is a path, e.g. a --out argument) and writes or updates its
    manifest.json: git commit, date, command, plus `info`.
    """
    path = name if os.sep in name or "/" in name else os.path.join(OUTPUTS, name)
    os.makedirs(path, exist_ok=True)
    manifest_path = os.path.join(path, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    manifest.update({"name": name, "git_commit": commit + ("-dirty" if dirty else ""), "date": time.strftime("%Y-%m-%d %H:%M"),
                     "command": " ".join([os.path.basename(sys.executable)] + sys.argv), **info})
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    return path
