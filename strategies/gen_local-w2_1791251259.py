from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 6
    percentile_window: int = 80
    percentile: float = 0.85
    hold_bars: int = 4

    def __post_init__(self):
        if self.trend_window < 2 or self.percentile_window < 2:
            raise ValueError("Windows must be at least two bars.")
        if not 0.0 < self.percentile < 1.0:
            raise ValueError("Percentile must lie strictly between zero and one.")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("Holding period must be three to five bars.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791251259"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(2 * params.trend_window + params.percentile_window + 3)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        t, w, q = params.trend_window, params.percentile_window, params.percentile
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(t, min_periods=t).mean().replace(0.0, np.nan)
        out["center"] = out["midpoint"].rolling(t, min_periods=t).mean()
        out["slope"] = out["center"].diff()
        out["expected_slope"] = out["slope"].shift(1).rolling(t, min_periods=t).mean()
        out["slope_innovation"] = (out["slope"] - out["expected_slope"]) / out["atr"]
        out["innovation_threshold"] = out["slope_innovation"].shift(1).rolling(w, min_periods=w).quantile(q)
        out["predicted_center"] = out["center"].shift(1) + out["slope"].shift(1)
        out["previous_atr"] = out["atr"].shift(1)
        out["upper_excursion"] = (data["high"] - out["predicted_center"]) / out["previous_atr"]
        out["excursion_threshold"] = out["upper_excursion"].shift(1).rolling(w, min_periods=w).quantile(q)
        out["upper_corridor"] = out["predicted_center"] + out["previous_atr"] * out["excursion_threshold"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        required = ["slope", "slope_innovation", "innovation_threshold", "upper_corridor"]
        ready = indicators[required].notna().all(axis=1) & data["close"].notna()
        entries = (ready
                   & (data["close"] > indicators["upper_corridor"])
                   & (indicators["slope"] > 0.0)
                   & (indicators["slope_innovation"] > 0.0)
                   & (indicators["slope_innovation"] > indicators["innovation_threshold"]))
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries.to_numpy(dtype=bool)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                    # Reserve this bar for exit; never renew an expiring position.
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    data = pd.DataFrame({"close": 2.0}, index=pd.date_range("2020-01-01", periods=14))
    indicators = pd.DataFrame({
        "slope": 1.0, "slope_innovation": 1.0,
        "innovation_threshold": 0.5, "upper_corridor": 1.0,
    }, index=data.index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    indicators.loc[indicators.index[1:], "slope_innovation"] = 0.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
