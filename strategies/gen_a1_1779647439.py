from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CoilOverlapParams:
    coil_window: int = 10
    coil_upper: float = 0.55
    coil_lower: float = 0.35
    breakout_lookback: int = 20
    confirm_bars: int = 2
    atr_period: int = 14
    breakeven_pct: float = 0.015
    trail_atr_mult: float = 2.0
    initial_stop_atr_mult: float = 2.0
    max_hold_bars: int = 5
    base_size: float = 1.0


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779647439"

    @classmethod
    def params_type(cls):
        return CoilOverlapParams

    @classmethod
    def warmup_bars(cls, params: CoilOverlapParams) -> int:
        return int(max(params.coil_window, params.breakout_lookback, params.atr_period)) + 5

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: CoilOverlapParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        bar_range = (high - low).clip(lower=1e-9)

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        overlap_top = pd.concat([high, prev_high], axis=1).min(axis=1)
        overlap_bot = pd.concat([low, prev_low], axis=1).max(axis=1)
        overlap = (overlap_top - overlap_bot).clip(lower=0.0)

        max_range = pd.concat([bar_range, bar_range.shift(1)], axis=1).max(axis=1)
        overlap_ratio = (overlap / max_range.clip(lower=1e-9)).fillna(0.0)

        coil_score = overlap_ratio.rolling(
            params.coil_window, min_periods=params.coil_window
        ).mean()

        tr1 = bar_range
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        coil_top = high.rolling(
            params.breakout_lookback, min_periods=params.breakout_lookback
        ).max()

        out = pd.DataFrame(index=data.index)
        out["bar_range"] = bar_range
        out["overlap_ratio"] = overlap_ratio
        out["coil_score"] = coil_score
        out["atr"] = atr
        out["coil_top"] = coil_top
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CoilOverlapParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        coil_score = indicators["coil_score"].to_numpy(dtype=float)
        coil_top = indicators["coil_top"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        raw_signal = np.zeros(n, dtype=np.int64)

        coil_regime = np.zeros(n, dtype=bool)
        in_coil = False
        for i in range(n):
            cs = coil_score[i]
            if np.isnan(cs):
                coil_regime[i] = False
                continue
            if in_coil:
                if cs < params.coil_lower:
                    in_coil = False
            else:
                if cs > params.coil_upper:
                    in_coil = True
            coil_regime[i] = in_coil

        coil_recent_window = max(params.confirm_bars + 1, 3)

        in_position = False
        entry_price = 0.0
        stop_price = 0.0
        breakeven_armed = False
        bars_in_trade = 0

        for i in range(n):
            ct = coil_top[i]
            a = atr[i]
            if np.isnan(ct) or np.isnan(a) or a <= 0.0:
                if in_position:
                    raw_signal[i] = 1
                continue

            if not in_position:
                if i < params.confirm_bars:
                    continue

                rail = coil_top[i - params.confirm_bars]
                if np.isnan(rail):
                    continue

                confirmed = True
                for j in range(params.confirm_bars):
                    if close[i - j] <= rail:
                        confirmed = False
                        break
                if not confirmed:
                    continue

                start = max(0, i - coil_recent_window - params.confirm_bars)
                if not coil_regime[start : i + 1].any():
                    continue

                in_position = True
                entry_price = close[i]
                stop_price = entry_price - params.initial_stop_atr_mult * a
                breakeven_armed = False
                bars_in_trade = 0
                raw_signal[i] = 1
            else:
                bars_in_trade += 1

                if (not breakeven_armed) and high[i] >= entry_price * (1.0 + params.breakeven_pct):
                    if entry_price > stop_price:
                        stop_price = entry_price
                    breakeven_armed = True

                if breakeven_armed:
                    new_trail = high[i] - params.trail_atr_mult * a
                    if new_trail > stop_price:
                        stop_price = new_trail

                exit_now = False
                if low[i] <= stop_price:
                    exit_now = True
                if bars_in_trade >= params.max_hold_bars:
                    exit_now = True

                if exit_now:
                    raw_signal[i] = 0
                    in_position = False
                    entry_price = 0.0
                    stop_price = 0.0
                    breakeven_armed = False
                    bars_in_trade = 0
                else:
                    raw_signal[i] = 1

        size_arr = np.full(n, float(params.base_size), dtype=float)

        df = pd.DataFrame(
            {
                "signal": raw_signal.astype(int),
                "size": size_arr,
            },
            index=data.index,
        )

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
