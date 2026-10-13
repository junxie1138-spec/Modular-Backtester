from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 6
    history_bars: int = 252
    neighbors: int = 8
    volume_bars: int = 20
    percentile_bars: int = 126
    entry_quantile: float = 0.75
    atr_bars: int = 14
    hold_bars: int = 18
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    cooldown_bars: int = 3


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1790986247"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(params.history_bars + params.shape_bars +
                   max(params.volume_bars, params.atr_bars) +
                   params.percentile_bars + params.hold_bars + 1)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        ret = close.pct_change(fill_method=None)
        scale = ret.rolling(params.volume_bars).std().replace(0, np.nan)
        volume_base = data["volume"].rolling(params.volume_bars).median().shift(1)
        out["return"] = ret
        out["return_scale"] = scale
        out["relative_volume"] = data["volume"] / volume_base.replace(0, np.nan)
        # Each shape preserves the order of returns and participation.
        for j in range(params.shape_bars):
            out[f"r{j}"] = (ret.shift(j) / scale).clip(-5, 5)
            out[f"v{j}"] = np.log(
                out["relative_volume"].where(out["relative_volume"] > 0)
            ).shift(j).clip(-3, 3)
        features = out[[f"{kind}{j}" for kind in ("r", "v")
                        for j in range(params.shape_bars)]]
        distances = []
        outcomes = []
        # Candidates are separated by one shape length. Their forward
        # outcomes end before the current shape begins.
        for lag in range(params.hold_bars + params.shape_bars,
                         params.history_bars + 1, params.shape_bars):
            delta = features - features.shift(lag)
            distance = delta.pow(2).sum(axis=1, min_count=features.shape[1])
            outcome = close.pct_change(params.hold_bars, fill_method=None).shift(
                lag - params.hold_bars
            )
            distances.append(distance.to_numpy())
            outcomes.append(outcome.to_numpy())
        if distances:
            distance_matrix = np.column_stack(distances)
            outcome_matrix = np.column_stack(outcomes)
            valid = np.isfinite(distance_matrix) & np.isfinite(outcome_matrix)
            ranked = np.argsort(np.where(valid, distance_matrix, np.inf), axis=1)
            count = min(params.neighbors, distance_matrix.shape[1])
            nearest = ranked[:, :count]
            chosen = np.take_along_axis(outcome_matrix, nearest, axis=1)
            chosen_valid = np.take_along_axis(valid, nearest, axis=1)
            forecast = np.where(chosen_valid, chosen, 0.0).sum(axis=1) / max(count, 1)
            forecast[chosen_valid.sum(axis=1) < params.neighbors] = np.nan
            out["forecast"] = forecast
        else:
            out["forecast"] = np.nan
        out["forecast_cut"] = out["forecast"].abs().rolling(
            params.percentile_bars
        ).quantile(params.entry_quantile).shift(1)
        out["volume_cut"] = out["relative_volume"].rolling(
            params.percentile_bars
        ).quantile(params.entry_quantile).shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0
        forecast = indicators["forecast"]
        qualified = (
            (forecast.abs() > indicators["forecast_cut"]) &
            (indicators["relative_volume"] > indicators["volume_cut"]) &
            (forecast * indicators["return"] > 0) &
            (indicators["atr"] > 0)
        ).fillna(False)
        candidates = np.where(qualified, np.sign(forecast.fillna(0)), 0).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        direction = 0
        pending = False
        entry = 0.0
        entry_bar = 0
        stop = -np.inf
        armed = False
        next_entry_bar = self.warmup_bars(params)
        for i in range(len(data)):
            if pending:
                pending = False
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    direction = 0
                    next_entry_bar = i + params.cooldown_bars + 1
                    continue
                entry = opens[i]
                entry_bar = i
                armed = False
                stop = direction * entry - params.trail_atr * atr[i - 1]
            if direction:
                # A single increasing stop in signed price coordinates
                # ratchets upward for longs and downward in price for shorts.
                adverse = lows[i] if direction == 1 else -highs[i]
                expired = i - entry_bar + 1 >= params.hold_bars
                breached = np.isfinite(adverse) and adverse <= stop
                if breached or expired:
                    direction = 0
                    next_entry_bar = i + params.cooldown_bars + 1
                elif np.isfinite(closes[i]) and np.isfinite(atr[i]):
                    gain = direction * (closes[i] / entry - 1.0)
                    armed = armed or gain >= params.breakeven_pct
                    if armed:
                        stop = max(stop, direction * entry,
                                   direction * closes[i] - params.trail_atr * atr[i])
                    if direction * closes[i] <= stop:
                        direction = 0
                        next_entry_bar = i + params.cooldown_bars + 1
            elif i >= next_entry_bar and candidates[i]:
                direction = int(candidates[i])
                pending = True
            raw[i] = direction
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
