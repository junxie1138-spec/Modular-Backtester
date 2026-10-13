from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DistributedUpsideParams:
    window: int = 4
    strength_threshold: float = 0.20
    position_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DistributedUpsideParams]):
    strategy_id = "gen_local_1790991695"

    @classmethod
    def params_type(cls) -> type[DistributedUpsideParams]:
        return DistributedUpsideParams

    @staticmethod
    def warmup_bars(params: DistributedUpsideParams) -> int:
        if not 2 <= params.window <= 9:
            raise ValueError("window must be between 2 and 9")
        return int(params.window + 1)

    def indicators(self, data: pd.DataFrame, params: DistributedUpsideParams) -> pd.DataFrame:
        self.warmup_bars(params)
        returns = data["close"].pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        window = params.window
        net = returns.rolling(window, min_periods=window).sum()
        gross = returns.abs().rolling(window, min_periods=window).sum()
        strongest_gain = returns.clip(lower=0.0).rolling(window, min_periods=window).max()
        residual_net = net - strongest_gain
        residual_gross = gross - strongest_gain
        strength = residual_net.div(residual_gross.where(residual_gross > 0.0))
        ready = net.notna() & gross.notna() & strongest_gain.notna()
        result = pd.DataFrame(index=data.index)
        result["return"] = returns.fillna(0.0)
        result["net_return"] = net.fillna(0.0)
        result["gross_return"] = gross.fillna(0.0)
        result["strongest_gain"] = strongest_gain.fillna(0.0)
        result["residual_net"] = residual_net.fillna(0.0)
        result["residual_gross"] = residual_gross.fillna(0.0)
        result["strength"] = strength.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["ready"] = ready
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DistributedUpsideParams,
    ) -> SignalFrame:
        if not np.isfinite(params.position_size) or params.position_size <= 0.0:
            raise ValueError("position_size must be positive and finite")
        if not np.isfinite(params.strength_threshold) or not 0.0 <= params.strength_threshold < 1.0:
            raise ValueError("strength_threshold must be in [0, 1)")
        entry_condition = (
            indicators["ready"]
            & (indicators["residual_net"] > 0.0)
            & (indicators["strength"] > params.strength_threshold)
        )
        df = pd.DataFrame(index=data.index)
        # Hold while the entry condition remains true; exit only when it flips false.
        df["signal"] = entry_condition.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.position_size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
