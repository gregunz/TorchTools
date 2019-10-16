from pathlib import Path

from .cnn_out_size import cnn_out_size
from .seed import set_seed


def get_incremental_path(root: Path, s: str) -> Path:
    i = 0
    while (root / s.format(i)).exists():
        i += 1
    return root / s.format(i)
