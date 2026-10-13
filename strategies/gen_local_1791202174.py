from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ElasticParams:
    spring_window: int = 4
    atr_window: int = 20
    expansion_ratio: float = 1.15
    tension_threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[ElasticParams]):
    strategy_id = "gen_local_1791202174"

    @classmethod
    def params_type(cls) -> type[ElasticParams]:
        return ElasticParams

    def warmup_bars(self, params: ElasticParams) -> int:
        return int(max(200, params.spring_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: ElasticParams) -> pd.DataFrame:
        if params.spring_window < 2 or params.atr_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not np.isfinite(params.expansion_ratio) or params.expansion_ratio <= 0:
            raise ValueError("Expansion ratio must be finite and positive")
        if not np.isfinite(params.tension_threshold) or params.tension_threshold <= 0:
            raise ValueError("Tension threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["fast_atr"] = out["true_range"].rolling(params.spring_window, min_periods=params.spring_window).mean()
        out["baseline_atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["anchor"] = data["close"].rolling(params.spring_window, min_periods=params.spring_window).mean()
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        denominator = out["baseline_atr"].where(out["baseline_atr"] > 0)
        out["volatility_ratio"] = out["fast_atr"] / denominator
        out["tension"] = (data["close"] - out["anchor"]) / denominator
        out["force"] = out["tension"] * np.where(out["volatility_ratio"] >= params.expansion_ratio, 1.0, -1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ElasticParams) -> SignalFrame:
        valid = np.isfinite(indicators[["force", "volatility_ratio", "ma200"]]).all(axis=1) & np.isfinite(data["close"])
        setup = pd.Series(0, index=data.index, dtype=int)
        setup.loc[valid & (indicators["force"] >= params.tension_threshold)] = 1
        setup.loc[valid & (indicators["force"] <= -params.tension_threshold)] = -1
        above = (data["close"] > indicators["ma200"]).to_numpy()
        below = (data["close"] < indicators["ma200"]).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(setup.to_numpy()):
            if position != 0:
                # Only an opposing directional setup exits; neutral bars retain exposure.
                if direction == -position:
                    position = 0
            elif direction == 1 and above[i]:
                position = 1
            elif direction == -1 and below[i]:
                position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    sample = pd.DataFrame({"close": [101., 101., 99., 99., 99., 101., 101.]}, index=index)
    derived = pd.DataFrame({"force": [1., 0., -1., -1., 0., 1., 0.], "volatility_ratio": 1., "ma200": 100.}, index=index)
    result = GeneratedStrategy().generate_signals(sample, derived, None, ElasticParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
