from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CenterwardParams:
    slope_window: int = 4
    channel_window: int = 5
    innovation_threshold: float = 0.15
    displacement_threshold: float = 0.25


class GeneratedStrategy(BaseStrategy[CenterwardParams]):
    strategy_id = "gen_local_1791227666"

    @classmethod
    def params_type(cls) -> type[CenterwardParams]:
        return CenterwardParams

    def warmup_bars(self, params: CenterwardParams) -> int:
        return int(max(max(2, params.slope_window) + 2,
                       max(2, params.channel_window) + 1))

    def indicators(self, data: pd.DataFrame, params: CenterwardParams) -> pd.DataFrame:
        n = max(2, int(params.slope_window))
        r = max(2, int(params.channel_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        denominator = float(np.dot(weights, weights))
        slope = pd.Series(0.0, index=data.index)
        for lag in range(n):
            slope = slope + weights[n - 1 - lag] * out["log_close"].shift(lag)
        out["slope"] = slope / denominator
        out["slope_forecast"] = 2.0 * out["slope"].shift(1) - out["slope"].shift(2)
        out["slope_innovation"] = out["slope"] - out["slope_forecast"]
        out["log_change"] = out["log_close"].diff()
        out["change_scale"] = out["log_change"].abs().rolling(n, min_periods=n).mean()
        out["innovation_score"] = out["slope_innovation"] / out["change_scale"].where(out["change_scale"] > 0)
        out["channel_high"] = data["high"].rolling(r, min_periods=r).max().shift(1)
        out["channel_low"] = data["low"].rolling(r, min_periods=r).min().shift(1)
        out["channel_center"] = (out["channel_high"] + out["channel_low"]) / 2.0
        out["channel_width"] = out["channel_high"] - out["channel_low"]
        out["relative_position"] = 2.0 * (close - out["channel_center"]) / out["channel_width"].where(out["channel_width"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CenterwardParams) -> SignalFrame:
        score = indicators["innovation_score"]
        position = indicators["relative_position"]
        valid = np.isfinite(score) & np.isfinite(position)
        threshold = max(0.0, float(params.innovation_threshold))
        displacement = max(0.0, float(params.displacement_threshold))
        long_entry = valid & (score > threshold) & (position < -displacement)
        short_entry = valid & (score < -threshold) & (position > displacement)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        held = 0
        for bar, entry in enumerate(entries):
            if held == 0:
                held = int(entry)
            elif entry == -held:
                held = 0
            signals[bar] = held
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({
        "innovation_score": [np.nan, 1.0, 0.0, -1.0, -1.0, np.nan, 1.0, 1.0],
        "relative_position": [np.nan, -1.0, 0.0, 1.0, 1.0, np.nan, -1.0, -1.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, CenterwardParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
