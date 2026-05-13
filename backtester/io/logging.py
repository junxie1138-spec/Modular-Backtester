from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger("backtester")
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file.setFormatter(fmt)
        root.addHandler(file)

    root.propagate = False
    return root
