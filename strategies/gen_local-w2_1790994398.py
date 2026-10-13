from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityBasinParams:
    visibility_window: int = 16
    min_remote_degree: int = 3
    atr_period: int = 14
    trail_atr: float = 1.2
    max_holding_days: int = 2
    max_basin_position: float = 0.35


class GeneratedStrategy(BaseStrategy[VisibilityBasinParams]):
    strategy_id = "gen_local-w2_1790994398"

    @classmethod
    def params_type(cls) -> type[VisibilityBasinParams]:
        return VisibilityBasinParams

    def warmup_bars(self, params: VisibilityBasinParams) -> int:
        return int(max(200, params.visibility_window + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityBasinParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        window = max(4, int(params.visibility_window))
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous.notna())
        out["atr"] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["channel_low"] = close.rolling(window, min_periods=window).min()
        out["channel_high"] = close.rolling(window, min_periods=window).max()
        span = (out["channel_high"] - out["channel_low"]).replace(0.0, np.nan)
        out["relative_position"] = (close - out["channel_low"]) / span

        # Horizontal visibility on inverse price: both endpoint closes must
        # lie below every intervening close. Count only nonlocal edges.
        blocker = previous.copy()
        remote_degree = pd.Series(0.0, index=data.index)
        remote_start = max(3, window // 3)
        for lag in range(2, window):
            past = close.shift(lag)
            visible = close.lt(blocker) & past.lt(blocker)
            if lag >= remote_start:
                remote_degree = remote_degree + visible.astype(float)
            blocker = pd.Series(np.minimum(blocker.to_numpy(), past.to_numpy()), index=data.index)
        complete = close.rolling(window, min_periods=window).count().eq(window)
        out["remote_degree"] = remote_degree.where(complete)
        out["prior_degree"] = out["remote_degree"].shift(1)
        out["earlier_degree"] = out["remote_degree"].shift(2)
        out["prior_position"] = out["relative_position"].shift(1)
        out["rebound"] = close.gt(previous).where(previous.notna())
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityBasinParams) -> SignalFrame:
        required = ["ma200", "atr", "prior_degree", "earlier_degree", "prior_position", "rebound"]
        ready = indicators[required].notna().all(axis=1)
        ready &= np.isfinite(data["close"]) & np.isfinite(indicators["atr"])
        ready &= indicators["atr"].gt(0.0)
        regime = data["close"].gt(indicators["ma200"])
        entry = (
            ready
            & regime
            & indicators["prior_degree"].ge(params.min_remote_degree)
            & indicators["prior_degree"].gt(indicators["earlier_degree"])
            & indicators["prior_position"].le(params.max_basin_position)
            & indicators["rebound"].fillna(False).astype(bool)
        )
        close_values = data["close"].to_numpy(dtype=float)
        atr_values = indicators["atr"].to_numpy(dtype=float)
        entry_values = entry.to_numpy(dtype=bool)
        valid_values = (ready & regime).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = np.nan
        held_days = 0
        holding_limit = max(1, min(2, int(params.max_holding_days)))

        for i in range(len(data)):
            if active:
                # The entry decision was made on the previous close; this
                # is the first actual holding day if high_water is unset.
                if not valid_values[i]:
                    active = False
                else:
                    held_days += 1
                    if np.isnan(high_water):
                        high_water = close_values[i]
                    else:
                        high_water = max(high_water, close_values[i])
                    candidate_stop = high_water - params.trail_atr * atr_values[i]
                    stop = candidate_stop if np.isnan(stop) else max(stop, candidate_stop)
                    if close_values[i] <= stop or held_days >= holding_limit:
                        active = False
                decisions[i] = int(active)
                if not active:
                    high_water = np.nan
                    stop = np.nan
                    held_days = 0
            elif entry_values[i]:
                active = True
                high_water = np.nan
                stop = np.nan
                held_days = 0
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
