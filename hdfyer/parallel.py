"""--parallel N: several chapter pipelines at once, one per worker process, for GPUs one Python process can't keep busy.

A single warm-up process first builds every TensorRT engine and batch-profile entry the to-do chapters need, so
the workers only ever read those caches. Workers share a queue of chapters; each loads hdfyer once.
"""

import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from hdfyer.ingest import reader
from hdfyer.ingest.library import todo_chapters
from hdfyer.ingest.reader import prefetched
from hdfyer.digest import engine
from hdfyer.digest.engine import engine_stats
from hdfyer.digest.strips import ChunkMeta
from hdfyer.pipeline import SCALE, STRIP_HEIGHT, load_runtime, print_run_summary, process_chapter

# Below this much VRAM one pipeline already keeps the GPU busy.
PARALLEL_MIN_VRAM_GB = 16


def configure_threads(parallel: int) -> dict:
    """
    Caps this process's thread pools at its share of the CPUs, max(1, cpu_count // parallel): torch's intra-op
    threads, OpenCV's threads and the decode pool. Every pool is only ever lowered (min with what it is now), so a
    single process (parallel=1) keeps the defaults. Without this, N processes each start a pool sized for the whole
    machine (torch: one thread per core) and the threads of all of them fight over the same cores.
    """
    share = max(1, (os.cpu_count() or 1) // parallel)
    torch.set_num_threads(min(torch.get_num_threads(), share))
    cv2.setNumThreads(min(cv2.getNumThreads(), share))
    reader.decode_threads = min(reader.DECODE_THREADS, share)
    return {"share": share, "torch": torch.get_num_threads(), "opencv": cv2.getNumThreads(), "decode": reader.decode_threads}


def _warm_up(weights: Path, backend: str, batch: str | int, cbz_paths: list[str], results) -> None:
    """
    Warm-up process, run before any worker: measures every missing batch-profile entry and builds every missing
    TensorRT engine for the page widths in the to-do chapters, so that no worker ever writes a cache file.
    """
    try:
        configure_threads(1)   # runs alone, before any worker
        model, backend_used, batch_size = load_runtime(weights, backend, batch)
        engine.set_context("warm-up")
        widths = sorted({w for path in cbz_paths for w, _ in reader.page_sizes(path)})
        if callable(batch_size):
            for width in widths:
                batch_size(STRIP_HEIGHT, width)
        # One standard strip per width: prepare_chapter builds whatever engine is missing, shrinking the batch on failure.
        strips = [ChunkMeta(0, np.zeros((STRIP_HEIGHT, width, 3), np.uint8), 0, 1, 0, STRIP_HEIGHT, 0, 0,
                            page_name=f"{width} px wide pages") for width in widths]
        engine.prepare_chapter(strips, SCALE)
        results.put({"backend": backend_used, "widths": widths, "builds": engine_stats()["build_seconds"],
                     "events": engine.fallback_events(), "state": engine.warm_up_state(),
                     "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3})
    except BaseException:
        results.put({"error": traceback.format_exc()})


def _worker(worker_id: int, weights: Path, backend: str, batch: str | int, state: dict, output_dir: Path,
            cpu_workers: int, max_inflight: int, overlap: bool, parallel: int, tasks, results) -> None:
    """One pipeline: loads hdfyer once, then processes chapters from `tasks` until it gets None."""
    # Ctrl+C is the parent's to handle: it stops the workers, so a half-done chapter is simply left as .part.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        threads = configure_threads(parallel)
        model, _, batch_size = load_runtime(weights, backend, batch, allow_build=False, measure=False)
        engine.adopt_warm_up(state["failed_shapes"], state["batch_override"])
        reported = len(engine.fallback_events())
        results.put({"type": "ready", "worker": worker_id, "events": engine.fallback_events(), "threads": threads})
        def queued():
            while (task := tasks.get()) is not None:
                yield Path(task[0]), Path(task[1])

        for (cbz_path, relative), pages in prefetched(queued()):
            engine.set_context(f"w{worker_id} {relative}")
            elapsed = process_chapter(cbz_path, relative, output_dir, model, batch_size, cpu_workers, max_inflight, overlap,
                                      quiet=True, pages=pages)
            del pages
            events = engine.fallback_events()
            results.put({"type": "done", "worker": worker_id, "relative": str(relative), "seconds": elapsed,
                         "events": events[reported:]})
            reported = len(events)
        results.put({"type": "exit", "worker": worker_id, "events": engine.fallback_events()[reported:],
                     "builds": engine_stats()["build_seconds"]})
    except BaseException:
        results.put({"type": "error", "worker": worker_id, "error": traceback.format_exc()})


def process_all_parallel(manwha_dir: Path, output_dir: Path, weights: Path, backend: str, batch: str | int,
                         cpu_workers: int, max_inflight: int, overlap: bool, parallel: int) -> None:
    """
    `parallel` worker processes share a queue of to-do chapters. A single warm-up process first fills the engine
    and batch-profile caches for every page width in those chapters, so the workers only ever read them.
    The parent process never touches the GPU.
    """
    ctx = mp.get_context("spawn")
    todo = todo_chapters(manwha_dir, output_dir)
    print(f"{len(todo)} chapters to do, {parallel} worker processes.")
    if not todo:
        return

    print("Warm-up: batch profiles and TensorRT engines for every page width in those chapters...")
    warm_results = ctx.Queue()
    warm = ctx.Process(target=_warm_up, args=(weights, backend, batch, [str(cbz) for cbz, _ in todo], warm_results))
    warm.start()
    try:
        warm_up = warm_results.get()
    finally:
        if warm.is_alive() and "warm_up" not in locals():
            warm.terminate()   # Ctrl+C during the warm-up: an engine being built is written to a private temp file, never half-saved
        warm.join()
    if "error" in warm_up:
        raise RuntimeError("warm-up failed:\n" + warm_up["error"])
    backend = warm_up["backend"]
    print(f"Warm-up done: widths {warm_up['widths']}, backend {backend}.")
    if warm_up["vram_gb"] < PARALLEL_MIN_VRAM_GB:
        print(f"WARNING: --parallel {parallel} on a GPU with {warm_up['vram_gb']:.0f} GB of VRAM. Below {PARALLEL_MIN_VRAM_GB} GB "
              f"one pipeline already keeps the GPU busy, so extra workers mostly compete for it and for memory.")

    tasks, results = ctx.Queue(), ctx.Queue()
    for cbz, relative in todo:
        tasks.put((str(cbz), str(relative)))
    for _ in range(parallel):
        tasks.put(None)
    workers = [ctx.Process(target=_worker, args=(i + 1, weights, backend, batch, warm_up["state"], output_dir,
                                                 cpu_workers, max_inflight, overlap, parallel, tasks, results))
               for i in range(parallel)]

    builds, events = list(warm_up["builds"]), list(warm_up["events"])
    completed, exited = 0, set()
    total_start = time.perf_counter()
    progress = tqdm(total=len(todo), desc="Processing chapters")
    try:
        for worker in workers:
            worker.start()
        while len(exited) < parallel:
            try:
                message = results.get(timeout=2)
            except queue.Empty:
                dead = [i + 1 for i, w in enumerate(workers) if not w.is_alive() and (i + 1) not in exited]
                if dead:
                    raise RuntimeError(f"worker(s) {dead} died without reporting (killed, or out of host memory?)")
                continue
            kind = message["type"]
            if kind == "ready":
                events += message["events"]
                if message["worker"] == 1:
                    t = message["threads"]
                    tqdm.write(f"Threads per worker (share of {os.cpu_count()} CPUs: {t['share']}): torch {t['torch']}, "
                               f"OpenCV {t['opencv']}, decode {t['decode']}, finishing {cpu_workers}.")
            elif kind == "done":
                completed += 1
                events += message["events"]
                progress.update(1)
                tqdm.write(f"[w{message['worker']}] Done: {message['relative']} in {message['seconds']:.1f}s")
            elif kind == "exit":
                exited.add(message["worker"])
                events += message["events"]
                builds += message["builds"]
            elif kind == "error":
                raise RuntimeError(f"worker {message['worker']} failed:\n{message['error']}")
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        tqdm.write("Interrupted: stopping the workers. Chapters they were on stay as .part files and are redone next run.")
        raise
    finally:
        progress.close()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()   # mid-chapter: its output is still a .part, never renamed, so never taken as finished
        for worker in workers:
            if worker.pid is not None:
                worker.join(timeout=30)
        for q in (tasks, results):
            q.cancel_join_thread()   # nobody will drain them now; don't block exit on their feeder threads
            q.close()
        total_elapsed = time.perf_counter() - total_start
        avg = total_elapsed / completed if completed else 0
        print(f"\nFinished {completed} chapters in {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min), avg {avg:.1f}s/chapter, "
              f"{parallel} workers.")
        print_run_summary(backend, builds, events)
