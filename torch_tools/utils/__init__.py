from pathlib import Path

from .cnn_out_size import cnn_out_size
from .load_dict import smart_load_state_dict
from .seed import set_seed
from .tensorboard_logs import TensorboardLogs


def get_incremental_path(root: Path, s: str) -> Path:
    i = 0
    while (root / s.format(i)).exists():
        i += 1
    return root / s.format(i)
