from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DepthReclaimParams:
    reference_bars: int = 63
    recovery_bars: int = 18
    smoothing_bars: int = 3
    minimum_depth: float = 0.03
    dominance_threshold: float = 0.25
    position_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DepthReclaimParams]):
    strategy_id = "gen_local-w2_1791320222"

    @classmethod
    def params_type(cls) -> type[DepthReclaimParams]:
        return DepthReclaimParams

    def warmup_bars(self, params: DepthReclaimParams) -> int:
        return int(max(2, params.reference_bars) + max(2, params.recovery_bars) + max(1, params.smoothing_bars) - 2)

    def indicators(self, data: pd.DataFrame, params: DepthReclaimParams) -> pd.DataFrame:
        reference = max(2, params.reference_bars)
        recovery = max(2, params.recovery_bars)
        smoothing = max(1, params.smoothing_bars)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["peak"] = out["log_close"].rolling(reference, min_periods=reference).max()
        out["trough"] = out["log_close"].rolling(reference, min_periods=reference).min()
        out["drawdown"] = out["peak"] - out["log_close"]
        out["drawup"] = out["log_close"] - out["trough"]
        out["maximum_drawdown"] = out["drawdown"].rolling(recovery, min_periods=recovery).max()
        out["maximum_drawup"] = out["drawup"].rolling(recovery, min_periods=recovery).max()
        out["downside_reclaim"] = 1.0 - out["drawdown"] / out["maximum_drawdown"].replace(0.0, np.nan)
        out["upside_reclaim"] = 1.0 - out["drawup"] / out["maximum_drawup"].replace(0.0, np.nan)
        out["reclaim_difference"] = out["downside_reclaim"] - out["upside_reclaim"]
        out["dominance"] = out["reclaim_difference"].rolling(smoothing, min_periods=smoothing).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DepthReclaimParams) -> SignalFrame:
        ready = indicators[["maximum_drawdown", "maximum_drawup", "dominance"]].notna().all(axis=1)
        deep = (indicators["maximum_drawdown"] >= params.minimum_depth) & (indicators["maximum_drawup"] >= params.minimum_depth)
        threshold = max(0.000001, float(params.dominance_threshold))
        long_entry = ready & deep & (indicators["dominance"] >= threshold)
        short_entry = ready & deep & (indicators["dominance"] <= -threshold)
        desired = np.select([long_entry.to_numpy(), short_entry.to_numpy()], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(desired):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        size = float(params.position_size)
        df["size"] = size if np.isfinite(size) and size > 0 else 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
