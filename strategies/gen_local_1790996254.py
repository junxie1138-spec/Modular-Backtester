from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class BoundaryParams:
    rail_window: int = 5
    motion_lag: int = 3
    baseline_window: int = 20
    atr_window: int = 14
    pressure_window: int = 3
    translation_atr: float = 0.25
    splay_ratio: float = 1.25
    dominance_ratio: float = 1.5
    pressure_threshold: float = 0.15
    breakeven_pct: float = 0.015
    initial_stop_atr: float = 2.0
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[BoundaryParams]):
    strategy_id = "gen_local_1790996254"

    @classmethod
    def params_type(cls) -> type[BoundaryParams]:
        return BoundaryParams

    def warmup_bars(self, params: BoundaryParams) -> int:
        return int(max(
            params.rail_window + params.baseline_window + params.motion_lag + 1,
            params.atr_window + 1,
            params.pressure_window,
        ))

    def indicators(self, data: pd.DataFrame, params: BoundaryParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float).clip(lower=0.0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        upper = high.rolling(params.rail_window, min_periods=params.rail_window).max()
        lower = low.rolling(params.rail_window, min_periods=params.rail_window).min()
        upper_motion = upper.diff(params.motion_lag)
        lower_motion = lower.diff(params.motion_lag)
        width = upper - lower
        baseline_width = width.rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).median().shift(1)
        expansion = width.div(baseline_width.where(baseline_width > 0.0))
        translation_up = (
            (upper_motion > params.translation_atr * atr)
            & (lower_motion > params.translation_atr * atr)
        )
        translation_down = (
            (upper_motion < -params.translation_atr * atr)
            & (lower_motion < -params.translation_atr * atr)
        )
        splaying = (
            (upper_motion > 0.0)
            & (lower_motion < 0.0)
            & (expansion >= params.splay_ratio)
        )
        upper_dominant = splaying & (
            upper_motion > params.dominance_ratio * lower_motion.abs()
        )
        lower_dominant = splaying & (
            lower_motion.abs() > params.dominance_ratio * upper_motion
        )
        range_direction = pd.Series(np.select(
            [translation_up, translation_down, upper_dominant, lower_dominant],
            [1, -1, -1, 1], default=0,
        ), index=data.index, dtype=int)
        daily_range = (high - low).where(high > low)
        closing_location = ((2.0 * close - high - low) / daily_range).clip(-1.0, 1.0).fillna(0.0)
        signed_volume = closing_location * volume
        pressure = signed_volume.rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum().div(volume.rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum().replace(0.0, np.nan))
        out = pd.DataFrame({
            "true_range": true_range,
            "atr": atr,
            "upper": upper,
            "lower": lower,
            "upper_motion": upper_motion,
            "lower_motion": lower_motion,
            "width": width,
            "baseline_width": baseline_width,
            "expansion": expansion,
            "closing_location": closing_location,
            "signed_volume": signed_volume,
            "pressure": pressure,
            "range_direction": range_direction,
            "translation_regime": (translation_up | translation_down).astype(int),
            "splay_regime": splaying.astype(int),
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: BoundaryParams,
    ) -> SignalFrame:
        rail_direction = indicators["range_direction"]
        pressure = indicators["pressure"]
        candidates = np.select([
            (rail_direction == 1) & (pressure >= params.pressure_threshold),
            (rail_direction == -1) & (pressure <= -params.pressure_threshold),
        ], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        entry = 0.0
        stop_floor = 0.0
        peak_profit = 0.0
        armed = False
        held = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            exited = False
            valid = (
                np.isfinite(opens[i]) and np.isfinite(highs[i])
                and np.isfinite(lows[i]) and np.isfinite(closes[i])
                and np.isfinite(atrs[i]) and opens[i] > 0.0
                and closes[i] > 0.0 and atrs[i] > 0.0
            )
            # A previous close's entry decision fills at this bar's open.
            if direction == 0 and i > 0 and decisions[i - 1] != 0:
                if valid:
                    direction = int(decisions[i - 1])
                    entry = opens[i]
                    stop_floor = -params.initial_stop_atr * atrs[i - 1]
                    peak_profit = 0.0
                    armed = False
                    held = 0
                else:
                    exited = True

            if direction != 0:
                held += 1
                if not valid:
                    direction = 0
                    exited = True
                else:
                    # Stops ratchet upward in directional profit coordinates.
                    # For shorts this corresponds to a decreasing price stop.
                    stop_price = entry + direction * stop_floor
                    touched = (
                        lows[i] <= stop_price if direction == 1
                        else highs[i] >= stop_price
                    )
                    profit = direction * (closes[i] - entry)
                    if touched:
                        direction = 0
                        exited = True
                    else:
                        peak_profit = max(peak_profit, profit)
                        if profit >= entry * params.breakeven_pct:
                            armed = True
                        if armed:
                            stop_floor = max(
                                stop_floor, 0.0,
                                peak_profit - params.trail_atr * atrs[i],
                            )
                        if profit <= stop_floor or held >= params.max_hold_bars:
                            direction = 0
                            exited = True

            if direction != 0:
                decisions[i] = direction
            elif not exited and valid and i >= warmup - 1:
                decisions[i] = int(candidates[i])

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        # Stop touches are decisions at the close, filled next bar, not at the stop price.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
