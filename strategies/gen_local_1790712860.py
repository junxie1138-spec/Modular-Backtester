from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    streak_window: int = 252
    analog_window: int = 504
    analog_band: float = 0.10
    entry_percentile: float = 0.80
    min_analogs: int = 8
    horizon: int = 15
    atr_window: int = 20
    breakeven_pct: float = 0.04
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712860"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.streak_window, params.analog_window) + params.horizon + params.atr_window + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        up = close.gt(close.shift(1)).fillna(False)
        groups = (~up).cumsum()
        streak = up.astype(int).groupby(groups).cumsum().astype(float)
        streak_pct = streak.rolling(params.streak_window, min_periods=params.streak_window).rank(pct=True)
        atr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1).rolling(params.atr_window, min_periods=params.atr_window).mean()

        # Outcome and feature values are aligned to when the analog outcome becomes known.
        forward_return = close.pct_change(params.horizon).shift(params.horizon)
        known_percentile = streak_pct.shift(params.horizon)
        return pd.DataFrame({
            "streak": streak,
            "streak_pct": streak_pct,
            "known_return": forward_return,
            "known_percentile": known_percentile,
            "atr": atr,
        }, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        pct = indicators["streak_pct"]
        hist_pct = indicators["known_percentile"]
        hist_ret = indicators["known_return"]
        similar = hist_pct.sub(pct, axis=0).abs().le(params.analog_band) & hist_ret.notna()
        analog_count = similar.astype(int).rolling(params.analog_window, min_periods=1).sum()
        analog_mean = hist_ret.where(similar).rolling(params.analog_window, min_periods=1).mean()
        entry = pct.ge(params.entry_percentile) & analog_count.ge(params.min_analogs) & analog_mean.gt(0)
        raw = entry.fillna(False).astype(int)

        # Hold for at most the target horizon; exit earlier on a breakeven-then-trail stop.
        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)
        in_trade = False
        entry_price = 0.0
        high_water = 0.0
        stop = 0.0
        entry_bar = -1
        atr = indicators["atr"].to_numpy()
        lows = data["low"].to_numpy()
        highs = data["high"].to_numpy()
        closes = data["close"].to_numpy()
        entries = raw.to_numpy()
        for i in range(len(data)):
            if in_trade:
                if lows[i] <= stop or i - entry_bar >= params.horizon:
                    in_trade = False
                    continue
                high_water = max(high_water, highs[i])
                if high_water >= entry_price * (1.0 + params.breakeven_pct):
                    stop = max(stop, entry_price, high_water - params.trail_atr * atr[i])
                signal[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                in_trade = True
                entry_price = closes[i]
                high_water = highs[i]
                stop = entry_price - params.trail_atr * atr[i]
                entry_bar = i
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
