from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapHandoffParams:
    regression_bars: int = 40
    shock_z: float = 1.0
    gap_miss_z: float = 0.6
    hold_bars: int = 4

    def __post_init__(self):
        if self.regression_bars < 3 or not 3 <= self.hold_bars <= 5:
            raise ValueError("Require regression_bars >= 3 and hold_bars between 3 and 5.")
        if not np.isfinite(self.shock_z) or not np.isfinite(self.gap_miss_z):
            raise ValueError("Thresholds must be finite.")
        if self.shock_z <= 0 or self.gap_miss_z <= 0:
            raise ValueError("Thresholds must be positive.")


class GeneratedStrategy(BaseStrategy[GapHandoffParams]):
    strategy_id = "gen_local-w2_1791266571"

    @classmethod
    def params_type(cls):
        return GapHandoffParams

    def warmup_bars(self, params):
        return max(200, params.regression_bars + 3)

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None)
        out["previous_return"] = out["return"].shift(1)
        out["gap"] = data["open"].where(data["open"] > 0) / close.shift(1) - 1.0
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        x = out["previous_return"]
        y = out["gap"]
        window = params.regression_bars
        out["x_mean"] = x.rolling(window, min_periods=window).mean().shift(1)
        out["gap_mean"] = y.rolling(window, min_periods=window).mean().shift(1)
        out["x_variance"] = x.rolling(window, min_periods=window).var().shift(1)
        out["covariance"] = x.rolling(window, min_periods=window).cov(y).shift(1)
        out["gap_std"] = y.rolling(window, min_periods=window).std().shift(1)
        out["return_std"] = out["return"].rolling(window, min_periods=window).std().shift(2)
        out["beta"] = out["covariance"] / out["x_variance"].where(out["x_variance"] > 0)
        out["expected_gap"] = out["gap_mean"] + out["beta"] * (x - out["x_mean"])
        out["gap_miss"] = (y - out["expected_gap"]) / out["gap_std"].where(out["gap_std"] > 0)
        out["shock"] = x / out["return_std"].where(out["return_std"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        long_entry = ready & (close > out["ma200"]) & (out["shock"] >= params.shock_z) & (out["gap_miss"] <= -params.gap_miss_z) & (out["return"] > 0)
        short_entry = ready & (close < out["ma200"]) & (out["shock"] <= -params.shock_z) & (out["gap_miss"] >= params.gap_miss_z) & (out["return"] < 0)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = GapHandoffParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1] + [-1] * 11}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, -1]
    assert result.index.equals(index) and (result["size"] > 0).all()
    flat = pd.DataFrame({name: 100.0 for name in ("open", "high", "low", "close", "volume")}, index=pd.date_range("2020-01-01", periods=260))
    assert (strategy.indicators(flat, params)["entry"] == 0).all()


if __name__ == "__main__":
    _self_check()
