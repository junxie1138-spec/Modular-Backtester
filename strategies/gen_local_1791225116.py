from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAbsorptionParams:
    gap_min: float = 0.002
    volume_mult: float = 1.15
    vol_lookback: int = 20
    atr_period: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 1.8
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[GapAbsorptionParams]):
    strategy_id = "gen_local_1791225116"

    @classmethod
    def params_type(cls) -> type[GapAbsorptionParams]:
        return GapAbsorptionParams

    @classmethod
    def warmup_bars(cls, params: GapAbsorptionParams) -> int:
        return max(params.vol_lookback, params.atr_period) + 1

    def indicators(self, data: pd.DataFrame, params: GapAbsorptionParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        previous_volume = data["volume"].shift(1).rolling(params.vol_lookback).mean()
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        result = pd.DataFrame(index=data.index)
        result["gap"] = (data["open"] / previous_close - 1).fillna(0.0)
        result["close_location"] = (
            (data["close"] - data["low"])
            / (data["high"] - data["low"]).replace(0, np.nan)
        ).fillna(0.0)
        result["volume_ratio"] = (
            data["volume"] / previous_volume.replace(0, np.nan)
        ).fillna(0.0)
        result["atr"] = true_range.rolling(params.atr_period).mean().fillna(0.0)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAbsorptionParams,
    ) -> SignalFrame:
        reversal = (
            (indicators["gap"] <= -params.gap_min)
            & (data["close"] > data["open"])
            & (indicators["close_location"] >= 0.6)
            & (indicators["volume_ratio"] >= params.volume_mult)
        )
        confirmation = (
            reversal.shift(1).fillna(False)
            & (data["open"] <= data["close"].shift(1))
            & (data["close"] > data["high"].shift(1))
            & (indicators["volume_ratio"] >= params.volume_mult)
        ).fillna(False)

        desired = np.zeros(len(data), dtype=int)
        active = False
        armed = False
        entry = 0.0
        peak = 0.0
        stop = float("-inf")
        held = 0
        for i in range(len(data)):
            if i and desired[i - 1] == 1 and not active:
                active = True
                armed = False
                entry = float(data["open"].iat[i])
                peak = entry
                stop = float("-inf")
                held = 0

            if active:
                held += 1
                low = float(data["low"].iat[i])
                close = float(data["close"].iat[i])
                hit_stop = low <= stop
                peak = max(peak, float(data["high"].iat[i]))
                if peak >= entry * (1 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * float(indicators["atr"].iat[i]))
                if hit_stop or (armed and close <= stop) or held >= params.max_hold:
                    active = False
                else:
                    desired[i] = 1
            elif bool(confirmation.iat[i]):
                desired[i] = 1

        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
