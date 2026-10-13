from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendInnovationParams:
    innovation_threshold: float = 0.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[TrendInnovationParams]):
    strategy_id = "gen_local_1790969536"

    @classmethod
    def params_type(cls) -> type[TrendInnovationParams]:
        return TrendInnovationParams

    @staticmethod
    def warmup_bars(params: TrendInnovationParams) -> int:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold < 0:
            raise ValueError("innovation_threshold must be finite and nonnegative")
        return 9

    @staticmethod
    def indicators(data: pd.DataFrame, params: TrendInnovationParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        out["log_return"] = out["log_close"].diff()
        out["noise"] = out["log_return"].abs().rolling(8, min_periods=8).mean()
        x = out["log_close"]
        # Four-observation least-squares slope, newest observation first.
        out["slope"] = 0.3 * x + 0.1 * x.shift(1) - 0.1 * x.shift(2) - 0.3 * x.shift(3)
        out["prior_slope"] = out["slope"].shift(4)
        out["innovation"] = ((out["slope"] - out["prior_slope"]) / out["noise"].where(out["noise"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["upper"] = data["high"].shift(1).rolling(8, min_periods=8).max()
        out["lower"] = data["low"].shift(1).rolling(8, min_periods=8).min()
        out["ready"] = np.isfinite(out[["slope", "prior_slope", "innovation", "upper", "lower"]]).all(axis=1) & np.isfinite(close)
        long_entry = out["ready"] & (close > out["upper"]) & (out["slope"] > 0) & (out["innovation"] > params.innovation_threshold)
        short_entry = out["ready"] & (close < out["lower"]) & (out["slope"] < 0) & (out["innovation"] < -params.innovation_threshold)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TrendInnovationParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = TrendInnovationParams(hold_bars=5)
    index = pd.date_range("2020-01-01", periods=13)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) == 9


if __name__ == "__main__":
    _self_check()
