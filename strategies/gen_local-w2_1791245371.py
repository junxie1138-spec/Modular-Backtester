from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    fast_window: int = 5
    slow_window: int = 20
    feedback_window: int = 30
    entry_strength: float = 0.2
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791245371"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.fast_window + 2, params.slow_window + 1,
                       params.feedback_window + 2, 4))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["return"] = close.pct_change(fill_method=None)
        out["magnitude"] = out["return"].abs()
        out["lagged_magnitude"] = out["magnitude"].shift(1)
        out["fast_vol"] = out["return"].rolling(params.fast_window).std(ddof=0)
        out["slow_vol"] = out["return"].rolling(params.slow_window).std(ddof=0)
        denominator = out["slow_vol"].replace(0.0, np.nan)
        out["volatility_ratio"] = out["fast_vol"] / denominator
        out["prior_volatility_ratio"] = out["volatility_ratio"].shift(1)
        out["feedback"] = out["magnitude"].rolling(params.feedback_window).corr(
            out["lagged_magnitude"]
        ).clip(-1.0, 1.0)
        out["impulse"] = close.pct_change(3, fill_method=None) / (denominator * np.sqrt(3.0))
        out["prior_shock"] = -out["return"].shift(1) / denominator
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.fast_window).mean()

        clustered = (
            (out["feedback"] > 0.15)
            & (out["volatility_ratio"] > 1.0)
            & (out["impulse"] > 0.0)
            & (out["return"] > 0.0)
        )
        alternating = (
            (out["feedback"] < -0.15)
            & (out["prior_shock"] > 1.0)
            & (out["volatility_ratio"] < out["prior_volatility_ratio"])
            & (out["impulse"] < 0.0)
            & (out["return"] > 0.0)
        )
        out["clustered_score"] = (
            ((out["feedback"] - 0.15) / 0.5).clip(0.0, 1.0)
            * out["impulse"].clip(0.0, 1.0)
        ).where(clustered, 0.0)
        out["alternating_score"] = (
            ((-out["feedback"] - 0.15) / 0.5).clip(0.0, 1.0)
            * (out["prior_shock"] / 2.0).clip(0.0, 1.0)
        ).where(alternating, 0.0)
        out["strength"] = out[["clustered_score", "alternating_score"]].max(axis=1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["ready"] = out.notna().all(axis=1) & (out["atr"] > 0.0) & (close > 0.0)
        out["entry"] = out["ready"] & (out["strength"] >= params.entry_strength)
        return out.fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        strengths = indicators["strength"].to_numpy(dtype=float)
        active = False
        age = 0
        highest = np.nan
        stop = -np.inf
        position_size = 1.0

        for i in range(n):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                age += 1
                if valid:
                    # Entry fills on this bar; exclude the preceding decision close.
                    highest = close if np.isnan(highest) else max(highest, close)
                    stop = max(stop, highest - params.trail_atr * atr)
                if not valid or close <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signal[i] = 1
                    sizes[i] = position_size
            elif valid and entries[i] and np.isfinite(strengths[i]):
                active = True
                age = 0
                highest = np.nan
                stop = -np.inf
                position_size = float(0.2 + 0.75 * np.clip(strengths[i], 0.0, 1.0))
                signal[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(trail_atr=1.0)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": [100.0, 102.0, 103.0, 101.0, 104.0, 105.0, 106.0, 107.0]}, index=index)
    indicators = pd.DataFrame(
        {"atr": [1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0],
         "entry": [True, False, False, False, False, False, False, False],
         "strength": [0.5] * 8}, index=index,
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert np.isclose(result["size"].iloc[1], 0.575)
    data["close"] = 100.0
    indicators["atr"] = 1.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    indicators["strength"] = 1.0
    larger = strategy.generate_signals(data, indicators, None, params).data
    assert larger["size"].iloc[1] > result["size"].iloc[1]


if __name__ == "__main__":
    _self_check()
