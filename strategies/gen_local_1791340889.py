from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAnalogParams:
    atr_period: int = 14
    history: int = 126
    gap_quantile: float = 0.25
    neighbor_fraction: float = 0.10
    trail_atr: float = 0.8
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[GapAnalogParams]):
    strategy_id = "gen_local_1791340889"

    @classmethod
    def params_type(cls) -> type[GapAnalogParams]:
        return GapAnalogParams

    @staticmethod
    def warmup_bars(params: GapAnalogParams) -> int:
        return int(params.history + params.atr_period + 6)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapAnalogParams) -> pd.DataFrame:
        if params.atr_period < 1 or params.history < 5:
            raise ValueError("atr_period must be positive and history at least five")
        if not 0 < params.gap_quantile < 1:
            raise ValueError("gap_quantile must be between zero and one")
        if not 0 < params.neighbor_fraction <= 1:
            raise ValueError("neighbor_fraction must be in (0, 1]")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        if params.max_hold not in (1, 2):
            raise ValueError("max_hold must be one or two sessions")

        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["prior_atr"] = out["atr"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] - prior_close) / out["prior_atr"]
        out["gap_lag1"] = out["gap"].shift(1)
        out["gap_lag2"] = out["gap"].shift(2)
        out["repair"] = (data["close"] - data["open"]) / out["prior_atr"]
        out["gap_threshold"] = out["gap"].shift(1).rolling(
            params.history, min_periods=params.history
        ).quantile(params.gap_quantile)

        features = out[["gap", "gap_lag1", "gap_lag2", "repair"]].replace(
            [np.inf, -np.inf], np.nan
        )
        distances = pd.DataFrame(index=data.index)
        outcomes = pd.DataFrame(index=data.index)
        for age in range(4, params.history + 4):
            difference = features - features.shift(age)
            distances[age] = difference.pow(2).sum(axis=1, min_count=4)
            # Historical signal at h buys at h+1 open and exits at h+hold+1 open.
            outcomes[age] = (
                data["open"].shift(age - params.max_hold - 1)
                / data["open"].shift(age - 1).where(lambda s: s > 0)
                - 1.0
            )
        distances = distances.replace([np.inf, -np.inf], np.nan)
        outcomes = outcomes.replace([np.inf, -np.inf], np.nan)
        out["distance_cutoff"] = distances.quantile(params.neighbor_fraction, axis=1)
        selected = distances.le(out["distance_cutoff"], axis=0) & outcomes.notna()
        out["neighbor_count"] = selected.sum(axis=1)
        out["forecast"] = outcomes.where(selected).mean(axis=1)
        out["positive_fraction"] = outcomes.gt(0).astype(float).where(selected).mean(axis=1)
        out["entry"] = (
            out["gap"].lt(0)
            & out["gap"].le(out["gap_threshold"])
            & out["repair"].gt(0)
            & out["forecast"].gt(0)
            & out["positive_fraction"].gt(0.5)
            & out["neighbor_count"].ge(3)
            & features.notna().all(axis=1)
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAnalogParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        held = 0
        highest_close = np.nan
        stop = -np.inf
        warmup = GeneratedStrategy.warmup_bars(params)

        for i in range(len(data)):
            if active:
                held += 1
                close = closes[i]
                atr = atrs[i]
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    active = False
                else:
                    # First held close occurs after the next-open entry fill.
                    highest_close = close if held == 1 else max(highest_close, close)
                    stop = max(stop, highest_close - params.trail_atr * atr)
                    if close < stop or held >= params.max_hold:
                        active = False
                signals[i] = int(active)
            elif i >= warmup - 1 and entries[i]:
                active = True
                held = 0
                highest_close = np.nan
                stop = -np.inf
                signals[i] = 1

        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
