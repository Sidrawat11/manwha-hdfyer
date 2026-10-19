"""Runs every test group; replaces the old `python test.py`. None of it needs a GPU or the model.

    python -m tests.run_all [path/to/chapter.cbz]   (real-page round trip; defaults to the first CBZ in data/library/)
"""

import sys
from pathlib import Path
from tests import test_digest, test_ingest, test_pipeline, test_render

GROUPS = [("ingest", test_ingest.checks), ("digest", test_digest.checks), ("render", test_render.checks),
          ("pipeline", test_pipeline.checks)]


def main() -> int:
    cbz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    totals, failed_all = 0, 0
    for name, run in GROUPS:
        print(f"{name}:")
        results = run(cbz_path) if run is test_digest.checks else run()
        failed = [description for description, ok in results if not ok]
        for description in failed:
            print(f"  FAIL {description}")
        print(f"  {'OK  ' if not failed else 'FAIL'} {len(results) - len(failed)}/{len(results)} {name} checks")
        totals += len(results)
        failed_all += len(failed)
    print(f"\n{totals - failed_all}/{totals} passed")
    return 1 if failed_all else 0


if __name__ == "__main__":
    sys.exit(main())
