from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransitionParams:
    baseline_window: int = 20
    recent_window: int = 4
    atr_window: int = 14
    shift_threshold: float = 0.65
    trailing_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapTransitionParams]):
    strategy_id = "gen_local_1790987979"

    @classmethod
    def params_type(cls) -> type[GapTransitionParams]:
        return GapTransitionParams

    def warmup_bars(self, params: GapTransitionParams) -> int:
        return int(max(params.baseline_window + params.recent_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapTransitionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = (data["open"] / out["prior_close"].replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["down_gap"] = out["gap"].lt(0.0).astype(float).where(out["gap"].notna())
        out["prior_gap"] = out["gap"].shift(params.recent_window)
        out["prior_down_gap"] = out["down_gap"].shift(params.recent_window)
        out["baseline_mean"] = out["prior_gap"].rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["baseline_std"] = out["prior_gap"].rolling(params.baseline_window, min_periods=params.baseline_window).std(ddof=0)
        out["baseline_down_frequency"] = out["prior_down_gap"].rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["recent_mean"] = out["gap"].rolling(params.recent_window, min_periods=params.recent_window).mean()
        out["recent_down_frequency"] = out["down_gap"].rolling(params.recent_window, min_periods=params.recent_window).mean()
        out["gap_shift"] = ((out["recent_mean"] - out["baseline_mean"]) / out["baseline_std"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["transition"] = (
            out["gap_shift"].ge(params.shift_threshold)
            & out["recent_mean"].gt(0.0)
            & out["baseline_down_frequency"].ge(0.55)
            & out["recent_down_frequency"].le(0.25)
        ).fillna(False)
        out["transition_event"] = out["transition"] & ~out["transition"].shift(1, fill_value=False)
        out["confirmation"] = (
            data["close"].gt(out["prior_close"])
            & data["close"].ge(data["open"])
        ).fillna(False)
        out["entry"] = (
            out["transition_event"].shift(2, fill_value=False)
            & out["confirmation"].shift(1, fill_value=False)
            & out["confirmation"]
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapTransitionParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        peak = -np.inf
        stop = -np.inf
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0.0:
                active = False
                age = 0
                peak = -np.inf
                stop = -np.inf
                continue

            if active:
                age += 1
                peak = max(peak, closes[i])
                stop = max(stop, peak - params.trailing_atr * atrs[i])
                if closes[i] <= stop or age >= params.max_hold_bars:
                    active = False
                    age = 0
                    peak = -np.inf
                    stop = -np.inf
            elif i >= warmup - 1 and entries[i]:
                active = True
                age = 0
                # The next bar is the fill bar; its close starts the in-trade high-water mark.
                peak = -np.inf
                stop = -np.inf

            raw[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
