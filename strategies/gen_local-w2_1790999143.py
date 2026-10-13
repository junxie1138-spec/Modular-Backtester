from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionTensionParams:
    peak_window: int = 20
    range_window: int = 10
    overlap_window: int = 4
    depth_threshold: float = 1.2
    migration_threshold: float = 0.05


class GeneratedStrategy(BaseStrategy[AuctionTensionParams]):
    strategy_id = "gen_local-w2_1790999143"

    @classmethod
    def params_type(cls) -> type[AuctionTensionParams]:
        return AuctionTensionParams

    def warmup_bars(self, params: AuctionTensionParams) -> int:
        return int(max(params.peak_window, params.range_window + 1, params.overlap_window + 1))

    def indicators(self, data: pd.DataFrame, params: AuctionTensionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out["range_scale"] = out["true_range"].rolling(
            params.range_window, min_periods=params.range_window
        ).mean().replace(0.0, np.nan)
        out["peak"] = data["close"].rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out["drawdown_depth"] = (out["peak"] - data["close"]) / out["range_scale"]
        out["overlap_floor"] = data["low"].rolling(
            params.overlap_window, min_periods=params.overlap_window
        ).max()
        out["overlap_ceiling"] = data["high"].rolling(
            params.overlap_window, min_periods=params.overlap_window
        ).min()
        overlapping = out["overlap_ceiling"] > out["overlap_floor"]
        out["accepted_midpoint"] = (
            (out["overlap_floor"] + out["overlap_ceiling"]) / 2.0
        ).where(overlapping)
        out["migration"] = out["accepted_midpoint"].diff() / out["range_scale"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AuctionTensionParams,
    ) -> SignalFrame:
        valid = indicators[["drawdown_depth", "migration"]].notna().all(axis=1)
        long_entry = (
            valid
            & (indicators["drawdown_depth"] >= params.depth_threshold)
            & (indicators["migration"] > params.migration_threshold)
        )
        short_entry = (
            valid
            & (indicators["drawdown_depth"] < params.depth_threshold)
            & (indicators["migration"] < -params.migration_threshold)
        )
        longs = long_entry.to_numpy(dtype=bool)
        shorts = short_entry.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i in range(len(data)):
            if position == 1:
                if not longs[i]:
                    position = 0
            elif position == -1:
                if not shorts[i]:
                    position = 0
            elif longs[i]:
                position = 1
            elif shorts[i]:
                position = -1
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
