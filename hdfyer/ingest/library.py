"""The library: which chapters still need doing."""

from pathlib import Path
from tqdm import tqdm


def todo_chapters(manwha_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """(cbz, relative path) of every chapter without a finished output. A leftover .part does not count as finished."""
    todo = []
    for cbz_path in sorted(manwha_dir.rglob("*.cbz")):
        relative = cbz_path.relative_to(manwha_dir)
        if (output_dir / relative).exists():
            tqdm.write(f"Skipping, already processed: {relative}")
            continue
        todo.append((cbz_path, relative))
    return todo
