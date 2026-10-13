from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    trend_span: int = 12
    peak_window: int = 63
    innovation_window: int = 3
    atr_window: int = 14
    min_drawdown: float = 0.02
    max_drawdown: float = 0.12
    turn_days: float = 3.0
    min_repair_atr: float = 0.04
    initial_stop_atr: float = 2.0
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790985382"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window, params.atr_window + 1,
                       2 * params.trend_span + params.innovation_window + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window,
                                 min_periods=params.atr_window).mean()
        first = close.ewm(span=params.trend_span, adjust=False,
                          min_periods=params.trend_span).mean()
        second = first.ewm(span=params.trend_span, adjust=False,
                           min_periods=params.trend_span).mean()
        alpha = 2.0 / (params.trend_span + 1.0)
        level = 2.0 * first - second
        slope = alpha / max(1.0 - alpha, 1e-12) * (first - second)
        innovation = slope.diff()
        repair = innovation.rolling(params.innovation_window,
                                    min_periods=params.innovation_window).mean()
        repair_atr = repair / atr.replace(0.0, np.nan)
        turn_time = (-slope) / repair.where(repair > 0.0)
        peak = close.rolling(params.peak_window,
                             min_periods=params.peak_window).max()
        drawdown = 1.0 - close / peak.replace(0.0, np.nan)
        ready = pd.Series(np.arange(len(data)) >= self.warmup_bars(params) - 1,
                          index=data.index)
        entry_ok = (
            ready
            & (atr > 0.0)
            & (drawdown >= params.min_drawdown)
            & (drawdown <= params.max_drawdown)
            & (slope < 0.0)
            & (innovation > 0.0)
            & (repair_atr >= params.min_repair_atr)
            & (turn_time > 0.0)
            & (turn_time <= params.turn_days)
            & (close > previous)
        )
        strength = (repair_atr / max(3.0 * params.min_repair_atr, 1e-12)).clip(0.0, 1.0)
        urgency = (1.0 - turn_time / max(params.turn_days, 1e-12)).clip(0.0, 1.0)
        confidence = (0.6 * strength.fillna(0.0) + 0.4 * urgency.fillna(0.0)).clip(0.0, 1.0)
        result = pd.DataFrame({
            "atr": atr,
            "first_smooth": first,
            "second_smooth": second,
            "latent_level": level,
            "latent_slope": slope,
            "slope_innovation": innovation,
            "slope_repair": repair,
            "repair_atr": repair_atr,
            "turn_time": turn_time,
            "rolling_peak": peak,
            "drawdown": drawdown,
            "confidence": confidence,
            "exposure": 0.2 + 0.8 * confidence,
            "entry_ok": entry_ok.fillna(False),
        }, index=data.index)
        numeric = result.columns.drop("entry_ok")
        result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.2, dtype=float)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(n):
            if pending:
                # The previous close's entry decision fills at this open.
                entry = float(data["open"].iloc[i])
                atr_previous = float(indicators["atr"].iloc[i - 1])
                stop = entry - params.initial_stop_atr * atr_previous
                peak = entry
                age = 0
                armed = False
                pending = False
                active = True
            if active:
                age += 1
                close = float(data["close"].iloc[i])
                low = float(data["low"].iloc[i])
                # Test the existing stop before updating it at this close.
                # An exit decision also executes on the following bar.
                if low <= stop or age >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, close)
                if close >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry,
                               peak - params.trail_atr * float(indicators["atr"].iloc[i]))
                signal[i] = 1
                size[i] = float(indicators["exposure"].iloc[i])
            elif bool(indicators["entry_ok"].iloc[i]) and i + 1 < n:
                signal[i] = 1
                size[i] = float(indicators["exposure"].iloc[i])
                pending = True
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
