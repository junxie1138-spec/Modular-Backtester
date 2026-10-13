from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AssimilationParams:
    slope_span: int = 6
    noise_window: int = 20
    atr_window: int = 14
    min_strength: float = 0.20
    min_innovation: float = 0.05
    trail_atr: float = 1.2
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AssimilationParams]):
    strategy_id = "gen_local-w2_1791257859"

    @classmethod
    def params_type(cls):
        return AssimilationParams

    def warmup_bars(self, params: AssimilationParams) -> int:
        return int(max(params.slope_span, params.noise_window, params.atr_window) + 4)

    def indicators(self, data: pd.DataFrame, params: AssimilationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["change"] = close.diff()
        out["slope"] = out["change"].ewm(
            span=params.slope_span, adjust=False, min_periods=params.slope_span
        ).mean()
        out["innovation"] = out["change"] - out["slope"].shift(1)
        out["noise"] = out["change"].rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).replace(0.0, np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().replace(0.0, np.nan)
        out["strength"] = out["slope"] / out["noise"]
        out["surprise"] = out["innovation"] / out["noise"]
        out["slope_change"] = out["slope"].diff()
        out["decaying"] = (
            out["innovation"].abs() < out["innovation"].shift(1).abs()
        )
        valid = np.isfinite(out[["strength", "surprise", "atr", "slope_change"]]).all(axis=1)
        out["long_ready"] = (
            valid & out["decaying"]
            & (out["strength"] > params.min_strength)
            & (out["surprise"] > params.min_innovation)
            & (out["slope_change"] > 0.0)
        )
        out["short_ready"] = (
            valid & out["decaying"]
            & (out["strength"] < -params.min_strength)
            & (out["surprise"] < -params.min_innovation)
            & (out["slope_change"] < 0.0)
        )
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AssimilationParams,
    ) -> SignalFrame:
        long_ready = indicators["long_ready"].fillna(False).astype(bool)
        short_ready = indicators["short_ready"].fillna(False).astype(bool)
        long_entry = (long_ready & long_ready.shift(1, fill_value=False)).to_numpy()
        short_entry = (short_ready & short_ready.shift(1, fill_value=False)).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest_close = lowest_close = stop = np.nan
        hold_limit = min(2, max(1, int(params.max_hold_bars)))

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0.0:
                position = 0
                age = 0
                continue

            if position:
                age += 1
                if position == 1:
                    highest_close = max(highest_close, close)
                    stop = max(stop, highest_close - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    lowest_close = min(lowest_close, close)
                    stop = min(stop, lowest_close + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or age >= hold_limit:
                    position = 0
                    age = 0
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                age = 0
                highest_close = lowest_close = close
                stop = close - position * params.trail_atr * atr

            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10)
    prices = np.array([100, 101, 102, 100, 99, 98, 97, 99, 100, 100], dtype=float)
    sample = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        "atr": 1.0,
        "long_ready": [True, True, False, False, False, False, False, False, False, False],
        "short_ready": [False, False, False, False, True, True, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        None, sample, derived, None, AssimilationParams(trail_atr=1.0)
    )
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0, -1, -1, 0, 0]
    assert (result.data["size"] > 0.0).all()
