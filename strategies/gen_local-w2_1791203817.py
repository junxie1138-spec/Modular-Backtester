from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    ma_window: int = 20
    percentile_window: int = 126
    breakout_percentile: float = 0.80
    reservoir_window: int = 8
    reservoir_min: float = 0.50
    atr_window: int = 14
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local-w2_1791203817"

    @classmethod
    def params_type(cls) -> type[BreakoutParams]:
        return BreakoutParams

    @staticmethod
    def warmup_bars(params: BreakoutParams) -> int:
        return int(max(
            params.ma_window + params.percentile_window + 1,
            params.ma_window + params.reservoir_window + 3,
            params.atr_window + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["close_std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = ((close - out["ma"]) / out["close_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["threshold"] = out["z"].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.breakout_percentile)
        out["below_ma"] = out["z"].le(0.0).astype(float).where(out["z"].notna())
        out["reservoir"] = out["below_ma"].shift(3).rolling(params.reservoir_window, min_periods=params.reservoir_window).mean()
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["entry"] = (
            out["z"].gt(out["threshold"])
            & out["z"].shift(1).le(out["threshold"].shift(1))
            & out["z"].gt(out["z"].shift(1))
            & out["z"].shift(1).gt(out["z"].shift(2))
            & out["z"].shift(2).gt(0.0)
            & out["reservoir"].ge(params.reservoir_min)
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BreakoutParams,
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if active:
                age += 1
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0.0:
                    active = False
                else:
                    # First active close follows the entry decision and next-open fill.
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    if close <= stop or age >= params.hold_bars:
                        active = False
                signal[i] = int(active)
            elif entries[i] and np.isfinite(close) and np.isfinite(atr) and atr > 0.0:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
