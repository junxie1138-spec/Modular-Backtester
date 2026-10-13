from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    lookback: int = 12
    innovation_threshold: float = 1.0


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local_1790901215"

    @classmethod
    def params_type(cls) -> type[BreakoutParams]:
        return BreakoutParams

    def warmup_bars(self, params: BreakoutParams) -> int:
        return max(3, int(params.lookback)) + 2

    def indicators(self, data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        n = max(3, int(params.lookback))
        threshold = float(params.innovation_threshold)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("innovation_threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        out["bar"] = np.arange(len(data), dtype=float)
        out["range_center"] = (high + low) / 2.0
        out["log_close"] = np.log(close.where(close > 0))
        out["log_center"] = np.log(out["range_center"].where(out["range_center"] > 0))
        out["time_variance"] = out["bar"].rolling(n, min_periods=n).var()
        out["close_slope"] = out["log_close"].rolling(n, min_periods=n).cov(out["bar"]) / out["time_variance"]
        out["center_slope"] = out["log_center"].rolling(n, min_periods=n).cov(out["bar"]) / out["time_variance"]
        out["log_return"] = out["log_close"].diff()
        out["prior_volatility"] = out["log_return"].rolling(n, min_periods=n).std().shift(1)
        out["innovation_scale"] = (out["prior_volatility"] * (6.0 / (n * (n + 1)))).where(out["prior_volatility"] > 0)
        out["close_innovation"] = out["close_slope"].diff() / out["innovation_scale"]
        out["center_innovation"] = out["center_slope"].diff() / out["innovation_scale"]
        out["prior_high"] = high.rolling(n, min_periods=n).max().shift(1)
        out["prior_low"] = low.rolling(n, min_periods=n).min().shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        required = ["close_slope", "center_slope", "close_innovation", "center_innovation", "prior_high", "prior_low"]
        ready = out[required].notna().all(axis=1) & close.notna() & high.notna() & low.notna()
        out["long_setup"] = ready & (close > out["prior_high"]) & (out["close_slope"] > 0) & (out["center_slope"] > 0) & (out["close_innovation"] >= threshold) & (out["center_innovation"] >= threshold)
        out["short_setup"] = ready & (close < out["prior_low"]) & (out["close_slope"] < 0) & (out["center_slope"] < 0) & (out["close_innovation"] <= -threshold) & (out["center_innovation"] <= -threshold)
        out["entry"] = np.select([out["long_setup"], out["short_setup"]], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BreakoutParams) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = i + 4
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=11, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [1, -1, 0, 0, -1, -1, 1, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = BreakoutParams()
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert (result["size"] > 0).all()
    assert strategy.indicators(data, params)["entry"].eq(0).all()
    assert strategy.warmup_bars(params) >= params.lookback + 2
