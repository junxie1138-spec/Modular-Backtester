from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    peak_window: int = 40
    analog_history: int = 252
    shape_tolerance: float = 0.25
    min_forecast: float = 0.0025
    atr_window: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791303228"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return int(max(params.peak_window + params.analog_history + 6, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if params.peak_window < 2 or params.analog_history < 11 or params.atr_window < 1:
            raise ValueError("Invalid indicator lookback")
        if params.shape_tolerance <= 0 or params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError("Tolerance and exit parameters must be positive")

        close = data["close"]
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = 1.0 - close.div(peak.replace(0.0, np.nan))
        shape = pd.DataFrame({"depth": depth, "depth_2": depth.shift(2),
                              "depth_4": depth.shift(4), "depth_6": depth.shift(6)}, index=data.index)
        scale = shape.max(axis=1).clip(lower=0.01)
        trailing_return = close.div(close.shift(4).replace(0.0, np.nan)) - 1.0
        weighted_return = pd.Series(0.0, index=data.index)
        weighted_hit = pd.Series(0.0, index=data.index)
        weight_sum = pd.Series(0.0, index=data.index)
        matches = pd.Series(0, index=data.index, dtype=int)

        for lag in range(11, params.analog_history + 1):
            distance = ((shape - shape.shift(lag)).pow(2).mean(axis=1, skipna=False)).pow(0.5).div(scale)
            outcome = trailing_return.shift(lag - 4)
            eligible = distance.le(params.shape_tolerance) & outcome.notna()
            weight = np.exp(-distance.div(params.shape_tolerance)).where(eligible, 0.0).fillna(0.0)
            weighted_return += outcome.where(eligible, 0.0).fillna(0.0) * weight
            weighted_hit += outcome.gt(0.0).astype(float) * weight
            weight_sum += weight
            matches += eligible.astype(int)

        previous_close = close.shift(1)
        true_range = pd.concat([data["high"] - data["low"],
                                (data["high"] - previous_close).abs(),
                                (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out = shape.copy()
        out["peak"] = peak
        out["trailing_return_4"] = trailing_return
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["forecast"] = weighted_return.div(weight_sum.replace(0.0, np.nan)).fillna(0.0)
        out["hit_rate"] = weighted_hit.div(weight_sum.replace(0.0, np.nan)).fillna(0.0)
        out["matches"] = matches
        out["weight_sum"] = weight_sum
        out["candidate"] = (depth.between(0.01, 0.12) & depth.lt(depth.shift(1))
                            & out["forecast"].gt(params.min_forecast)
                            & out["hit_rate"].ge(0.65) & matches.ge(8)
                            & out["atr"].gt(0.0)).fillna(False)
        out["confirmed"] = out["candidate"] & out["candidate"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        pending = False
        active = False
        armed = False
        entry = 0.0
        stop = 0.0
        held = 0

        for i in range(len(data)):
            exited = False
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0 and i > 0 and np.isfinite(atrs[i - 1]):
                    entry = opens[i]
                    stop = entry - params.trail_atr * atrs[i - 1]
                    active = True
                    armed = False
                    held = 0

            if active:
                held += 1
                valid = np.isfinite(lows[i]) and np.isfinite(closes[i]) and np.isfinite(atrs[i])
                if not valid or lows[i] <= stop or held >= 4:
                    active = False
                    exited = True
                else:
                    previous_stop = stop
                    if closes[i] >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, closes[i] - params.trail_atr * atrs[i])
                    assert stop >= previous_stop
                    desired[i] = 1

            if not active and not exited and confirmed[i]:
                pending = True
                desired[i] = 1

        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
