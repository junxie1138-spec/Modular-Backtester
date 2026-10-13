from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResidualCorrectionParams:
    vol_window: int = 20
    atr_window: int = 14
    shock_z: float = 1.6
    correction_floor: float = 0.20
    correction_cap: float = 0.80
    trigger_pct: float = 0.005
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[ResidualCorrectionParams]):
    strategy_id = "gen_local-w2_1791298707"

    @classmethod
    def params_type(cls) -> type[ResidualCorrectionParams]:
        return ResidualCorrectionParams

    def warmup_bars(self, params: ResidualCorrectionParams) -> int:
        return max(params.vol_window + 4, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: ResidualCorrectionParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        shock = ret.shift(2).fillna(0.0)
        first = ret.shift(1).fillna(0.0)
        volatility = ret.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0).shift(3).fillna(0.0)
        corrected = -(first + ret) / shock.replace(0.0, np.nan)
        confirmed = (shock * first < 0) & (shock * ret < 0) & (ret.abs() < first.abs())
        qualified = confirmed & (shock.abs() >= params.shock_z * volatility) & (volatility > 0) & corrected.between(params.correction_floor, params.correction_cap)
        previous_close = close.shift(1)
        true_range = pd.concat([data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["candidate"] = np.where(qualified, -np.sign(shock), 0).astype(int)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ResidualCorrectionParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        position = 0
        entry = np.nan
        stop = np.nan
        armed = False
        held = 0
        for i in range(self.warmup_bars(params), len(data)):
            if position:
                if np.isnan(entry):
                    entry = float(data["open"].iat[i])
                held += 1
                high = float(data["high"].iat[i])
                low = float(data["low"].iat[i])
                close = float(data["close"].iat[i])
                if position == 1:
                    armed = armed or high >= entry * (1 + params.trigger_pct)
                    if armed:
                        stop = max(stop, entry, high - params.trail_atr * atr[i]) if np.isfinite(stop) else max(entry, high - params.trail_atr * atr[i])
                    stopped = armed and close <= stop
                else:
                    armed = armed or low <= entry * (1 - params.trigger_pct)
                    if armed:
                        stop = min(stop, entry, low + params.trail_atr * atr[i]) if np.isfinite(stop) else min(entry, low + params.trail_atr * atr[i])
                    stopped = armed and close >= stop
                if stopped or held >= 2:
                    position = 0
                    entry = np.nan
                    stop = np.nan
                    armed = False
                    held = 0
                signals[i] = position
            elif candidates[i]:
                position = int(candidates[i])
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
