from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAbsorptionParams:
    pressure_window: int = 15
    breakout_window: int = 20
    atr_window: int = 14
    minimum_absorbed_gaps: int = 3
    minimum_gap_atr: float = 0.35
    recovery_ratio: float = 0.80
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[GapAbsorptionParams]):
    strategy_id = "gen_local_1790994833"

    @classmethod
    def params_type(cls):
        return GapAbsorptionParams

    def warmup_bars(self, params: GapAbsorptionParams) -> int:
        return int(max(200, params.pressure_window + 2,
                       params.breakout_window + 1, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame,
                   params: GapAbsorptionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        gap = data["open"] - previous_close
        body = data["close"] - data["open"]
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["ma200"] = data["close"].rolling(200).mean()
        out["gap_atr"] = gap / out["prior_atr"].replace(0.0, np.nan)
        out["ceiling"] = data["close"].rolling(params.breakout_window).max().shift(1)
        out["floor"] = data["close"].rolling(params.breakout_window).min().shift(1)
        significant_down = out["gap_atr"] <= -params.minimum_gap_atr
        significant_up = out["gap_atr"] >= params.minimum_gap_atr
        out["down_absorbed"] = (
            significant_down & (body > 0.0)
        ).astype(float).rolling(params.pressure_window).sum().shift(1)
        out["up_absorbed"] = (
            significant_up & (body < 0.0)
        ).astype(float).rolling(params.pressure_window).sum().shift(1)
        out["down_gap_total"] = (-gap).where(
            significant_down, 0.0
        ).rolling(params.pressure_window).sum().shift(1)
        out["up_gap_total"] = gap.where(
            significant_up, 0.0
        ).rolling(params.pressure_window).sum().shift(1)
        out["down_body_total"] = body.where(
            significant_down, 0.0
        ).rolling(params.pressure_window).sum().shift(1)
        out["up_body_total"] = (-body).where(
            significant_up, 0.0
        ).rolling(params.pressure_window).sum().shift(1)
        out["down_recovery"] = out["down_body_total"] / out[
            "down_gap_total"
        ].replace(0.0, np.nan)
        out["up_recovery"] = out["up_body_total"] / out[
            "up_gap_total"
        ].replace(0.0, np.nan)
        # Only the direction's recovery statistic needs to exist.
        common_ready = out[
            ["atr", "prior_atr", "ma200", "gap_atr", "ceiling", "floor"]
        ].notna().all(axis=1) & (out["atr"] > 0.0) & (out["prior_atr"] > 0.0)
        long_entry = (
            common_ready
            & (data["close"] > out["ma200"])
            & (out["gap_atr"] >= params.minimum_gap_atr)
            & (data["open"] > out["ceiling"])
            & (data["close"] > out["ceiling"])
            & (out["down_absorbed"] >= params.minimum_absorbed_gaps)
            & (out["down_recovery"] >= params.recovery_ratio)
        )
        short_entry = (
            common_ready
            & (data["close"] < out["ma200"])
            & (out["gap_atr"] <= -params.minimum_gap_atr)
            & (data["open"] < out["floor"])
            & (data["close"] < out["floor"])
            & (out["up_absorbed"] >= params.minimum_absorbed_gaps)
            & (out["up_recovery"] >= params.recovery_ratio)
        )
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: GapAbsorptionParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        side = 0
        pending = 0
        entry_price = 0.0
        stop = 0.0
        best = 0.0
        armed = False
        held = 0
        for i in range(len(data)):
            if pending:
                # Yesterday's decision is filled at today's actual open.
                if np.isfinite(opens[i]) and opens[i] > 0.0 and i > 0 and np.isfinite(atrs[i - 1]):
                    side = pending
                    entry_price = opens[i]
                    best = side * entry_price
                    stop = best - params.initial_stop_atr * atrs[i - 1]
                    armed = False
                    held = 0
                pending = 0
            if side:
                held += 1
                valid_range = np.isfinite(highs[i]) and np.isfinite(lows[i])
                adverse = lows[i] if side == 1 else -highs[i]
                favorable = highs[i] if side == 1 else -lows[i]
                # Stop coordinates are signed prices: ratcheting upward
                # tightens a long sell stop or lowers a short buyback stop.
                # Test the previously established stop first. New stops
                # become effective next session; no intrabar ordering guess.
                exit_now = (valid_range and adverse <= stop) or held >= params.max_hold_bars
                if exit_now:
                    side = 0
                    continue
                if valid_range:
                    best = max(best, favorable)
                    if best - side * entry_price >= entry_price * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                            stop = max(stop, best - params.trail_atr * atrs[i])
                signals[i] = side
            elif entries[i]:
                pending = int(entries[i])
                signals[i] = pending
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
