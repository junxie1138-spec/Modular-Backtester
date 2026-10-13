from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapFloorParams:
    atr_period: int = 14
    vol_period: int = 20
    min_gap_atr: float = 0.6
    target_vol: float = 0.10
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapFloorParams]):
    strategy_id = "gen_local-w2_1790994251"

    @classmethod
    def params_type(cls):
        return GapFloorParams

    def warmup_bars(self, params: GapFloorParams) -> int:
        return int(max(params.atr_period + 2, params.vol_period + 1))

    def indicators(self, data: pd.DataFrame, params: GapFloorParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["gap_atr"] = out["gap"] / out["prior_atr"].where(out["prior_atr"] > 0)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["annual_vol"] = out["return"].rolling(
            params.vol_period, min_periods=params.vol_period
        ).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (
            params.target_vol / out["annual_vol"].clip(lower=0.01)
        ).replace([np.inf, -np.inf], np.nan).clip(lower=0.05, upper=0.95)
        # First confirmation: the gap session closes above its opening floor.
        out["first_defense"] = (
            (out["gap_atr"] <= -params.min_gap_atr)
            & (data["close"] > data["open"])
            & (data["close"] < out["prior_close"])
        ).fillna(False)
        out["gap_floor"] = data["open"].shift(1)
        out["gap_ceiling"] = out["prior_close"].shift(1)
        # Second confirmation: the next session preserves the floor and advances.
        out["confirmed"] = (
            out["first_defense"].shift(1, fill_value=False)
            & (data["low"] >= out["gap_floor"])
            & (data["close"] > out["prior_close"])
            & (data["close"] < out["gap_ceiling"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapFloorParams,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.95, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        target = indicators["target_size"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=bool)
        warmup = self.warmup_bars(params)
        pending = False
        active = False
        entry_atr = 0.0
        fixed_stop = 0.0
        position_size = 0.95
        held = 0

        for i in range(n):
            # Yesterday's entry decision fills at today's open. ATR was frozen
            # at the decision close; only today's observed open sets the stop.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                    held = 0

            if active:
                held += 1
                stopped = np.isfinite(closes[i]) and closes[i] < fixed_stop
                if stopped or held >= params.hold_bars:
                    active = False
                    # This zero decision exits at the following session's open.
                    signals[i] = 0
                else:
                    signals[i] = 1
                    sizes[i] = position_size
            elif (
                i >= warmup - 1
                and confirmed[i]
                and np.isfinite(atr[i])
                and atr[i] > 0
                and np.isfinite(target[i])
                and target[i] > 0
                and np.isfinite(closes[i])
                and closes[i] > 0
            ):
                pending = True
                entry_atr = float(atr[i])
                position_size = float(target[i])
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
