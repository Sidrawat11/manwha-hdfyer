"""Command line: upscale every chapter in Manwhas/ into results/ with hdfyer. The pipeline lives in hdfyer/."""

import argparse
import sys
from pathlib import Path
from hdfyer.pipeline import BACKEND, BACKENDS, BATCH, CPU_WORKERS, MAX_INFLIGHT, PARALLEL, WEIGHTS, process_all


def parse_batch(text: str) -> str | int:
    """--batch: "auto" or a whole number of strips, at least 1."""
    if text == "auto":
        return text
    try:
        batch = int(text)
    except ValueError:
        batch = 0
    if batch < 1:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a number of at least 1, got {text!r}")
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale every chapter in Manwhas/ into results/ with hdfyer.")
    parser.add_argument("--backend", choices=BACKENDS, default=BACKEND,
                        help="inference runtime: auto is TensorRT if installed, else eager (default: %(default)s)")
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="hdfyer weights (default: %(default)s)")
    parser.add_argument("--batch", type=parse_batch, default=BATCH,
                        help="strips per forward pass: auto uses the best batch measured for this GPU and page width, "
                             "profiling it on first use; a number overrides it (default: %(default)s)")
    parser.add_argument("--cpu-workers", type=int, default=CPU_WORKERS,
                        help="threads finishing pages (sharpen, cleanup, encode) while the GPU works (default: %(default)s)")
    parser.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT,
                        help="most upscaled pages queued or being finished at once, ~45 MB each (default: %(default)s)")
    parser.add_argument("--no-overlap", action="store_true",
                        help="finish each page on the main thread before the GPU continues (the old serial path)")
    parser.add_argument("--parallel", type=int, default=PARALLEL,
                        help="chapters processed at once, each in its own worker process; helps on large GPUs "
                             f"(16 GB+), not on laptops (default: %(default)s)")
    args = parser.parse_args()
    if args.cpu_workers < 1 or args.max_inflight < 1 or args.parallel < 1:
        parser.error("--cpu-workers, --max-inflight and --parallel must be at least 1")
    if not args.weights.exists():
        parser.error(f"hdfyer weights not found: {args.weights}")

    try:
        process_all(Path("Manwhas"), Path("results"), args.weights, backend=args.backend, batch=args.batch,
                    cpu_workers=args.cpu_workers, max_inflight=args.max_inflight, overlap=not args.no_overlap, parallel=args.parallel)
    except KeyboardInterrupt:
        # The summary has been printed; unfinished chapters are .part files and are redone on the next run.
        sys.exit(130)
