"""pipeline: the --parallel plumbing (no processes are started)."""

import json
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
import cv2
import numpy as np
import torch


def checks() -> list[tuple[str, bool]]:
    """--parallel plumbing: per-process thread pools, the warm-up's hand-over, workers never measuring."""
    from hdfyer import parallel
    from hdfyer.ingest import reader
    from hdfyer.digest import engine
    from hdfyer.digest import batch_profile as hdfyer_profile
    checks = []
    # Thread pools: a single process keeps its defaults; N processes each get max(1, cpu_count // N).
    saved_threads = (torch.get_num_threads(), cv2.getNumThreads(), reader.decode_threads)
    try:
        one = parallel.configure_threads(1)
        unchanged = (torch.get_num_threads(), cv2.getNumThreads(), reader.decode_threads) == saved_threads
        many = parallel.configure_threads(10 ** 6)
        share = (os.cpu_count() or 1) // 4
        torch.set_num_threads(saved_threads[0]); cv2.setNumThreads(saved_threads[1]); reader.decode_threads = saved_threads[2]
        four = parallel.configure_threads(4)
        checks.append(("threads: --parallel 1 keeps torch, OpenCV and decode threads as they were", unchanged and one["share"] == (os.cpu_count() or 1)))
        checks.append(("threads: N processes each get at most cpu_count // N, and never less than 1",
                       many == {"share": 1, "torch": 1, "opencv": 1, "decode": 1}
                       and four["torch"] <= max(1, share) and four["opencv"] <= max(1, share) and four["decode"] <= reader.DECODE_THREADS))
    finally:
        torch.set_num_threads(saved_threads[0]); cv2.setNumThreads(saved_threads[1]); reader.decode_threads = saved_threads[2]

    saved_runtime = engine._runtime
    try:
        engine._runtime = engine._Runtime()
        engine._runtime.failed_shapes, engine._runtime.batch_override = {(512, 1500)}, {(512, 800): 2}
        state = engine.warm_up_state()
        engine._runtime = engine._Runtime()
        engine.adopt_warm_up(state["failed_shapes"], state["batch_override"])
        checks.append(("parallel: workers start from the warm-up's failed shapes and lowered batches",
                       engine._runtime.failed_shapes == {(512, 1500)} and engine._runtime.batch_override == {(512, 800): 2}))
    finally:
        engine._runtime = saved_runtime

    saved = (hdfyer_profile.load_section, hdfyer_profile.profile_width)
    measured = []
    hdfyer_profile.load_section = lambda key: {"weights": "w", "strip_height": 512, "widths": {"800": {"best_batch": 2}}}
    hdfyer_profile.profile_width = lambda *args, **kwargs: measured.append(args) or {"best_batch": 9}
    try:
        profile = hdfyer_profile.BatchProfile(None, "w", 512, on_missing=lambda width: 1)
        checks.append(("parallel: a worker never measures a missing width; it answers through on_missing",
                       profile(512, 800) == 2 and profile(512, 1500) == 1 and measured == []))
    finally:
        hdfyer_profile.load_section, hdfyer_profile.profile_width = saved
    return checks
