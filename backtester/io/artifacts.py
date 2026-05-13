from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from backtester.config.loader import dump_run_config
from backtester.config.models import RunConfig
from backtester.core.types import BacktestResult
from backtester.io.serialization import write_json

PathLike = Union[str, Path]


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


class ArtifactWriter:
    def __init__(self, root: PathLike, run_name: str, now: Optional[str] = None):
        self.root = Path(root)
        self.now = now or _stamp()
        self.run_dir = self.root / f"{self.now}_{run_name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_config(self, rc: RunConfig) -> Path:
        path = self.run_dir / "config_resolved.yaml"
        dump_run_config(rc, path)
        return path

    def write_result(self, result: BacktestResult) -> None:
        write_json(self.run_dir / "summary.json", result.summary)
        result.trades.to_csv(self.run_dir / "trades.csv", index=False)
        result.positions.to_csv(self.run_dir / "positions.csv", index_label="timestamp")
        result.equity_curve.to_csv(self.run_dir / "equity_curve.csv", index_label="timestamp")

    def write_window_results(self, payload: Any) -> Path:
        path = self.run_dir / "window_results.json"
        write_json(path, payload)
        return path

    def log_path(self) -> Path:
        return self.run_dir / "logs.txt"
