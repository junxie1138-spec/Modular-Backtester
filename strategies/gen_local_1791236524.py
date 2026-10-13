from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 20
    history_bars: int = 126
    smooth_bars: int = 15


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791236524"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(2 * params.history_bars + 2 * params.shape_bars + params.smooth_bars + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        n, h, s = params.shape_bars, params.history_bars, params.smooth_bars
        if n < 4 or h < 10 or s < 1:
            raise ValueError("shape_bars >= 4, history_bars >= 10, smooth_bars >= 1 required")
        half = n // 2
        x = pd.DataFrame(index=data.index)
        x["volume_input"] = data["volume"].clip(lower=0).replace([np.inf, -np.inf], np.nan)
        x["range_input"] = ((data["high"] - data["low"]).clip(lower=0) / data["close"].where(data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        for feature in ("volume", "range"):
            source = x[feature + "_input"]
            x[feature + "_late"] = source.rolling(half, min_periods=half).mean()
            x[feature + "_early"] = source.rolling(n - half, min_periods=n - half).mean().shift(half)
            total = x[feature + "_late"] + x[feature + "_early"]
            x[feature + "_shape"] = x[feature + "_late"] / total.where(total > 0)
            x[feature + "_rank"] = x[feature + "_shape"].rolling(h, min_periods=h).rank(pct=True)
            x["past_" + feature] = x[feature + "_rank"].shift(n)
        x["formation_return"] = data["close"] / data["close"].shift(n).where(data["close"].shift(n) > 0) - 1.0
        x["past_return"] = x["formation_return"].shift(n).replace([np.inf, -np.inf], np.nan)
        x["past_valid"] = x[["past_volume", "past_range", "past_return"]].notna().all(axis=1)
        x["history_count"] = x["past_valid"].astype(float).rolling(h, min_periods=h).sum()
        for label, direction in (("bull", 1), ("bear", -1)):
            x[label + "_member"] = x["past_valid"] & (direction * x["past_return"] > 0)
            weight = x[label + "_member"].astype(float)
            x[label + "_count"] = weight.rolling(h, min_periods=h).sum()
            denominator = x[label + "_count"].where(x[label + "_count"] >= 5)
            for feature in ("volume", "range"):
                x[label + "_" + feature] = (x["past_" + feature].fillna(0.0) * weight).rolling(h, min_periods=h).sum() / denominator
            x[label + "_distance"] = sum(
                (x[feature + "_rank"] - x[label + "_" + feature]) ** 2
                for feature in ("volume", "range")
            )
        x["raw_score"] = (x["bear_distance"] - x["bull_distance"]).where(x["history_count"] == h)
        x["score"] = x["raw_score"].rolling(s, min_periods=s).mean()
        return x.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # One boundary: bullish analog wins => enter; it ceases winning => exit.
        df["signal"] = indicators["score"].gt(0.0).astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = AnalogParams()
    count = strategy.warmup_bars(params) + 100
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, count)))
    spread = rng.uniform(0.002, 0.025, count)
    data = pd.DataFrame({
        "open": close,
        "high": close * (1.0 + spread),
        "low": close * (1.0 - spread),
        "close": close,
        "volume": rng.lognormal(16.0, 0.4, count),
    }, index=pd.date_range("2000-01-01", periods=count, freq="B"))
    full = strategy.indicators(data, params)
    cut = count - 17
    prefix = strategy.indicators(data.iloc[:cut], params)
    assert np.allclose(full.iloc[:cut].to_numpy(dtype=float), prefix.to_numpy(dtype=float), equal_nan=True)
    assert full["score"].iloc[strategy.warmup_bars(params):].notna().all()
    scores = pd.DataFrame({"score": [np.nan, 0.2, 0.1, -0.2, 0.0, 0.3]}, index=data.index[:6])
    signals = strategy.generate_signals(data.iloc[:6], scores, None, params).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert signals["size"].gt(0).all()
    mirror = scores.copy()
    mirror["score"] = -mirror["score"]
    mirrored = strategy.generate_signals(data.iloc[:6], mirror, None, params).data
    assert mirrored["signal"].tolist() == [0, 0, 0, 0, 1, 0]


if __name__ == "__main__":
    _self_check()
