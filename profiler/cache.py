"""Disk persistence for the benchmark map.

Reads and writes GPU-keyed JSON files. Checks for an existing cache on startup
and skips profiling if the current GPU's map is already present.
"""

import os
from pathlib import Path
import torch
import json
from datetime import datetime

CACHE_DIR = Path("cache")  # folder where benchmark JSON files are stored

def _get_gpu_slug() -> str:
    """
    Returns a filename-safe slug for the current GPU.
    Lowercases the full device name, strips 'NVIDIA' and 'GeForce',
    strips extra whitespace, replaces remaining spaces with hyphens.
    Example: 'rtx-4060-laptop-gpu'
    """
    device_name = torch.cuda.get_device_name(0)
    name = device_name.replace("NVIDIA", "").replace("GeForce", "").strip()
    return name.lower().replace(" ", "-")


def cache_check() -> bool:
    """Returns True if a benchmark cache file exists for the current GPU, False otherwise."""
    device_name = _get_gpu_slug() + ".json"
    path = CACHE_DIR / device_name
    return path.exists()


def load_cache() -> dict:
    """
    Loads and returns the benchmark map for the current GPU from disk as a dict.
    Assumes cache_check() returned True before this is called.
    """
    device_name = _get_gpu_slug() + ".json"
    path = CACHE_DIR / device_name
    with open(path, 'r') as file:
        cache = json.load(file)
    return cache


def save_cache(data: dict) -> bool:
    """
    Writes the benchmark map dict to disk as JSON for the current GPU.
    Returns True on success, False on failure.
    """
    device_name = _get_gpu_slug() 
    path = CACHE_DIR / (device_name + ".json")
    data["gpu_model"] = device_name
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    data["profiled_at"] = formatted_now
    data["available_vram"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, data)
        return True
    except Exception as e:
        print(f"Failed to save: {e}")
        return False


def load_section(key: str) -> dict:
    """Returns one top-level section of the current GPU's cache file, or {} if the file or the section is missing."""
    path = CACHE_DIR / (_get_gpu_slug() + ".json")
    if not path.exists():
        return {}
    with open(path, 'r') as file:
        return json.load(file).get(key, {})


def save_section(key: str, section: dict) -> None:
    """Writes one top-level section into the current GPU's cache file, leaving everything else in it untouched."""
    slug = _get_gpu_slug()
    path = CACHE_DIR / (slug + ".json")
    data = {}
    if path.exists():
        with open(path, 'r') as file:
            data = json.load(file)
    data.setdefault("gpu_model", slug)
    data.setdefault("available_vram", torch.cuda.get_device_properties(0).total_memory / 1024**2)
    data[key] = section
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, data)


def _write_atomic(path: Path, data: dict) -> None:
    """Writes JSON to a temp file private to this process, then renames it over `path` (atomic on one filesystem),
    so a reader never sees a half-written file and concurrent writers never share a temp file."""
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    with open(part, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(part, path)
