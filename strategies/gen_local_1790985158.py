from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalDrawdownParams:
    peak_window: int = 21
    atr_window: int = 14
    month_start_weekdays: int = 4
    min_depth_atr: float = 1.0
    max_depth_atr: float = 4.0
    breakeven_pct: float = 0.008
    trail_atr: float = 1.5
    initial_stop_atr: float = 2.5
    max_hold_days: int = 4
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[SeasonalDrawdownParams]):
    strategy_id = "gen_local_1790985158"

    @classmethod
    def params_type(cls):
        return SeasonalDrawdownParams

    def warmup_bars(self, params: SeasonalDrawdownParams) -> int:
        return int(max(params.peak_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: SeasonalDrawdownParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["prior_peak"] = data["high"].rolling(
            params.peak_window, min_periods=params.peak_window
        ).max().shift(1)
        out["drawdown"] = (out["prior_peak"] - data["close"]).clip(lower=0.0)
        out["depth_atr"] = out["drawdown"] / out["atr"].where(out["atr"] > 0.0)
        days = np.asarray(data.index.date, dtype="datetime64[D]")
        month_starts = days.astype("datetime64[M]").astype("datetime64[D]")
        out["month_weekday"] = np.busday_count(month_starts, days) + 1
        out["bullish_body"] = data["close"] > data["open"]
        out["eligible"] = (
            (out["month_weekday"] <= params.month_start_weekdays)
            & (out["depth_atr"] >= params.min_depth_atr)
            & (out["depth_atr"] <= params.max_depth_atr)
            & out["bullish_body"]
            & (out["atr"] > 0.0)
        ).fillna(False)
        denominator = max(params.max_depth_atr - params.min_depth_atr, 1e-12)
        out["strength"] = (
            (out["depth_atr"] - params.min_depth_atr) / denominator
        ).clip(0.0, 1.0).fillna(0.0)
        out["target_size"] = (
            params.min_size + (params.max_size - params.min_size) * out["strength"]
        ).clip(lower=1e-6).fillna(max(params.min_size, 1e-6))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalDrawdownParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, max(params.min_size, 1e-6), dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        targets = indicators["target_size"].to_numpy(dtype=float)
        active = False
        pending_entry = False
        armed = False
        entry = 0.0
        stop = 0.0
        entry_atr = 0.0
        high_close = 0.0
        held = 0
        position_size = max(params.min_size, 1e-6)
        warmup = self.warmup_bars(params)

        for i in range(count):
            if active:
                # The preceding close's decision fills at this bar's open.
                if pending_entry:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        active = False
                        pending_entry = False
                        continue
                    entry = opens[i]
                    stop = entry - params.initial_stop_atr * entry_atr
                    high_close = entry
                    pending_entry = False
                held += 1
                valid_bar = (
                    np.isfinite(closes[i])
                    and np.isfinite(highs[i])
                    and np.isfinite(lows[i])
                    and np.isfinite(atrs[i])
                    and atrs[i] > 0.0
                )
                # Breaches are close-time exit decisions, filled next open.
                if not valid_bar or lows[i] <= stop or held >= params.max_hold_days:
                    active = False
                    continue
                high_close = max(high_close, closes[i])
                if highs[i] >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    # Ratchet at the close; never compare a newly raised stop
                    # against an earlier intraday low with unknown ordering.
                    stop = max(stop, entry, high_close - params.trail_atr * atrs[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                signals[i] = 1
                sizes[i] = position_size
            elif i >= warmup - 1 and eligible[i] and np.isfinite(targets[i]):
                active = True
                pending_entry = True
                armed = False
                held = 0
                entry_atr = atrs[i]
                position_size = max(targets[i], 1e-6)
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(max(params.min_size, 1e-6)).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
