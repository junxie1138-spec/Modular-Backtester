from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BreakoutParams:
    channel_bars: int = 12
    streak_bars: int = 2
    pressure_bars: int = 3
    pressure_threshold: float = 0.25
    atr_bars: int = 14
    stop_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[BreakoutParams]):
    strategy_id = "gen_local_1790984193"

    @classmethod
    def params_type(cls) -> type[BreakoutParams]:
        return BreakoutParams

    @staticmethod
    def warmup_bars(params: BreakoutParams) -> int:
        return int(max(params.channel_bars + params.streak_bars, params.atr_bars + 1, params.pressure_bars))

    @staticmethod
    def indicators(data: pd.DataFrame, params: BreakoutParams) -> pd.DataFrame:
        if min(params.channel_bars, params.streak_bars, params.pressure_bars, params.atr_bars) < 1:
            raise ValueError("Lookbacks must be positive")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be positive and finite")
        if not np.isfinite(params.pressure_threshold) or not 0 < params.pressure_threshold <= 1:
            raise ValueError("pressure_threshold must be in (0, 1]")

        out = pd.DataFrame(index=data.index)
        out["upper"] = data["high"].rolling(params.channel_bars).max().shift(1)
        out["lower"] = data["low"].rolling(params.channel_bars).min().shift(1)
        up = data["close"].gt(out["upper"]) & out["upper"].notna()
        down = data["close"].lt(out["lower"]) & out["lower"].notna()
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()
        out["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum()

        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        span = (data["high"] - data["low"]).replace(0, np.nan)
        out["close_location"] = ((2 * data["close"] - data["high"] - data["low"]) / span).clip(-1, 1).fillna(0)
        volume = data["volume"].clip(lower=0)
        out["signed_volume"] = out["close_location"] * volume
        out["volume_sum"] = volume.rolling(params.pressure_bars).sum()
        out["pressure"] = out["signed_volume"].rolling(params.pressure_bars).sum().div(out["volume_sum"].replace(0, np.nan))
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: BreakoutParams) -> SignalFrame:
        long_entry = indicators["up_streak"].ge(params.streak_bars) & indicators["pressure"].ge(params.pressure_threshold)
        short_entry = indicators["down_streak"].ge(params.streak_bars) & indicators["pressure"].le(-params.pressure_threshold)
        eligible = indicators["atr"].gt(0) & np.isfinite(indicators["atr"])
        candidate = np.select([long_entry & eligible, short_entry & eligible], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending_fill = False
        stop = np.nan
        entry_atr = np.nan
        age = 0
        warmup = GeneratedStrategy.warmup_bars(params)

        for i in range(len(data)):
            if pending_fill:
                pending_fill = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    position = 0
                else:
                    stop = opens[i] - position * params.stop_atr * entry_atr
                    age = 0

            exited = False
            if position:
                age += 1
                stop_hit = np.isfinite(closes[i]) and position * (closes[i] - stop) < 0
                if stop_hit or age >= params.hold_bars or not np.isfinite(closes[i]):
                    position = 0
                    exited = True

            if not position and not exited and i >= warmup - 1 and candidate[i]:
                position = int(candidate[i])
                entry_atr = atr[i]
                pending_fill = True

            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
