from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 20
    atr_window: int = 14
    slope_window: int = 20
    memory_window: int = 8
    regime_slope: float = 0.003
    trend_z: float = 0.5
    recovery_z: float = -1.0
    extreme_z: float = -2.0
    breakeven_pct: float = 0.03
    atr_multiple: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791313097"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200 + params.slope_window, params.ma_window + params.memory_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["price_std"] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["regime_slope"] = out["ma200"].diff(params.slope_window) / out["ma200"].shift(params.slope_window).replace(0.0, np.nan)
        previous_close = close.shift(1)
        true_range = pd.concat([(data["high"] - data["low"]), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        negative = (out["z"] < 0.0).astype(float).where(out["z"].notna())
        out["negative_occupancy"] = negative.shift(1).rolling(params.memory_window, min_periods=params.memory_window).sum()
        out["prior_z_min"] = out["z"].shift(1).rolling(params.memory_window, min_periods=params.memory_window).min()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        z = indicators["z"]
        bullish = data["close"] > indicators["ma200"]
        rising = indicators["regime_slope"] > params.regime_slope
        trend_entry = rising & (z > params.trend_z) & (z.shift(1) <= params.trend_z) & (indicators["negative_occupancy"] >= params.memory_window / 2.0)
        plateau_entry = ~rising & (z > params.recovery_z) & (z.shift(1) <= params.recovery_z) & (indicators["prior_z_min"] <= params.extreme_z)
        ready = indicators.notna().all(axis=1) & (indicators["atr"] > 0.0)
        entries = (ready & bullish & (trend_entry | plateau_entry)).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = bullish.to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = peak = stop = np.nan
        held = 0

        for i in range(len(data)):
            if pending:
                entry = opens[i]
                if not np.isfinite(entry) or entry <= 0.0:
                    raise ValueError("Entry open must be finite and positive")
                peak = entry
                stop = entry - params.atr_multiple * atr[i - 1]
                held = 0
                armed = False
                pending = False

            if active:
                close = closes[i]
                if not np.isfinite(close) or close <= 0.0 or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    raise ValueError("Active position requires valid close and ATR")
                held += 1
                peak = max(peak, close)
                if close >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.atr_multiple * atr[i])
                if close <= stop or not regime[i] or held >= params.holding_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and i + 1 < len(data):
                active = True
                pending = True
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
