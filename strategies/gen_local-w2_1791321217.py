from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    baseline_window: int = 40
    threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791321217"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if params.baseline_window < 5:
            raise ValueError("baseline_window must be at least 5")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        return int(params.baseline_window + 5)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["range_fraction"] = (
            (data["high"] - data["low"]) / data["close"].where(data["close"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["range_fraction"] = out["range_fraction"].where(out["range_fraction"] >= 0)
        out["recent_range"] = out["range_fraction"].rolling(5, min_periods=5).mean()
        out["baseline_range"] = out["range_fraction"].shift(5).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out["half_month"] = np.where(data.index.day <= 15, 1.0, -1.0)
        out["range_change"] = (
            out["recent_range"] / out["baseline_range"].where(out["baseline_range"] > 0) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        out["score"] = out["half_month"] * out["range_change"]
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        values = indicators["score"].to_numpy(dtype=float)
        desired = np.where(
            np.isfinite(values) & (values >= params.threshold), 1,
            np.where(np.isfinite(values) & (values <= -params.threshold), -1, 0),
        )
        raw = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(desired):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Exit to flat; a persistent opposite condition can enter next bar.
                position = 0
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"score": [np.nan, 0.2, 0.0, -0.2, -0.2, np.nan, 0.2, 0.2]}, index=index
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    self_check()
