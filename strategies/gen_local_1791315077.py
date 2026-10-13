from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    slope_window: int = 5
    burst_window: int = 3
    scale_window: int = 20
    channel_window: int = 8
    threshold: float = 0.75


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791315077"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.slope_window + params.scale_window + params.burst_window + params.channel_window + 2))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.slope_window, params.scale_window, params.channel_window) < 2 or params.burst_window < 1:
            raise ValueError("Invalid window length")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        n = params.slope_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        denominator = float(np.square(weights).sum())
        out["latent_slope"] = sum(
            weights[j] * out["log_close"].shift(n - 1 - j)
            for j in range(n)
        ) / denominator
        out["innovation"] = out["latent_slope"].diff()
        out["innovation_scale"] = out["innovation"].rolling(
            params.scale_window, min_periods=params.scale_window
        ).std(ddof=0).shift(1).replace(0.0, np.nan)
        out["burst"] = out["innovation"].rolling(
            params.burst_window, min_periods=params.burst_window
        ).sum()
        out["score"] = out["burst"] / (
            out["innovation_scale"] * np.sqrt(float(params.burst_window))
        )
        out["upper"] = out["score"].rolling(
            params.channel_window, min_periods=params.channel_window
        ).max().shift(1)
        out["lower"] = out["score"].rolling(
            params.channel_window, min_periods=params.channel_window
        ).min().shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        score = indicators["score"]
        ready = np.isfinite(indicators[["score", "upper", "lower", "ma200"]]).all(axis=1)
        upward = ready & (score > params.threshold) & (score > indicators["upper"])
        downward = ready & (score < -params.threshold) & (score < indicators["lower"])
        # The MA gates new exposure; the opposing directional condition exits it.
        long_entry = upward & (data["close"] > indicators["ma200"])
        short_entry = downward & (data["close"] < indicators["ma200"])
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if position == 1:
                if downward.iloc[i]:
                    position = 0
            elif position == -1:
                if upward.iloc[i]:
                    position = 0
            elif long_entry.iloc[i]:
                position = 1
            elif short_entry.iloc[i]:
                position = -1
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 101.0, "high": 102.0, "low": 100.0,
                         "close": 101.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [2.0, 0.0, -2.0, -2.0, 0.0, 2.0],
                               "upper": 1.0, "lower": -1.0,
                               "ma200": [100.0, 100.0, 102.0, 102.0, 102.0, 100.0]}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert (result["size"] > 0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=240))
    derived = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, derived, None, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
