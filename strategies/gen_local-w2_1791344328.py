from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockContainmentParams:
    shock_multiple: float = 1.5
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[ShockContainmentParams]):
    strategy_id = "gen_local-w2_1791344328"

    @classmethod
    def params_type(cls):
        return ShockContainmentParams

    @staticmethod
    def warmup_bars(params: ShockContainmentParams) -> int:
        return 22

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockContainmentParams) -> pd.DataFrame:
        if not np.isfinite(params.shock_multiple) or params.shock_multiple <= 0:
            raise ValueError("shock_multiple must be finite and positive")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["previous_peak"] = data["close"].shift(1).rolling(20, min_periods=20).max()
        out["drawdown"] = out["previous_peak"] - data["close"]
        out["shock"] = (
            (out["range"] >= params.shock_multiple * out["previous_atr"])
            & (data["close"] < out["previous_close"])
            & (out["drawdown"] >= out["previous_atr"])
        ).fillna(False)
        out["previous_shock"] = out["shock"].shift(1, fill_value=False)
        out["previous_range"] = out["range"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["contained"] = (
            (data["high"] <= out["previous_high"])
            & (data["low"] >= out["previous_low"])
            & (out["range"] <= 0.65 * out["previous_range"])
            & (out["range"] > 0)
        ).fillna(False)
        out["close_location"] = (data["close"] - data["low"]) / out["range"].where(out["range"] > 0)
        out["setup"] = out["previous_shock"] & out["contained"] & out["atr"].gt(0)
        out["entry"] = np.select(
            [
                out["setup"] & out["close_location"].ge(0.75) & data["close"].gt(out["previous_close"]),
                out["setup"] & out["close_location"].le(0.25) & data["close"].lt(out["previous_close"]),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockContainmentParams,
    ) -> SignalFrame:
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = 0.0
        stop = 0.0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                age += 1
                if position == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.stop_atr * atr)
                    stopped = close <= stop
                else:
                    # Mirror the high-water trail for shorts using a low-water mark.
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.stop_atr * atr)
                    stopped = close >= stop
                if stopped or age >= 4:
                    position = 0
                    age = 0
                    # Keep an exit bar flat even if another setup appears.
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                age = 0
                extreme = close
                stop = close - position * params.stop_atr * atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = ShockContainmentParams(stop_atr=2.0)
    cases = [
        ([100, 102, 99, 99, 99, 99], 1, [0, 1, 1, 0, 0, 0]),
        ([100, 98, 101, 101, 101, 101], -1, [0, -1, -1, 0, 0, 0]),
        ([100, 100, 100, 100, 100, 100], 1, [0, 1, 1, 1, 1, 0]),
    ]
    for closes, direction, expected in cases:
        index = pd.date_range("2020-01-01", periods=len(closes))
        data = pd.DataFrame({"close": closes}, index=index)
        derived = pd.DataFrame({"atr": 1.0, "entry": [direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, params)
        assert result.data["signal"].tolist() == expected
        assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
