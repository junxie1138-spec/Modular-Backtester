from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownBrakingParams:
    peak_window: int = 20
    noise_window: int = 20
    shock_mult: float = 2.0
    braking_ratio: float = 0.35
    min_depth: float = 0.005


class GeneratedStrategy(BaseStrategy[DrawdownBrakingParams]):
    strategy_id = "gen_local_1790998577"

    @classmethod
    def params_type(cls) -> type[DrawdownBrakingParams]:
        return DrawdownBrakingParams

    def warmup_bars(self, params: DrawdownBrakingParams) -> int:
        return int(max(200, params.peak_window + params.noise_window + 2))

    def indicators(self, data: pd.DataFrame, params: DrawdownBrakingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["peak"] = data["close"].rolling(params.peak_window, min_periods=params.peak_window).max()
        out["depth"] = (1.0 - data["close"] / out["peak"].replace(0.0, np.nan)).clip(lower=0.0)
        out["depth_change"] = out["depth"].diff()
        out["prior_change"] = out["depth_change"].shift(1)
        # Exclude both the shock bar and the current reaction bar from its baseline.
        out["noise"] = out["depth_change"].abs().rolling(params.noise_window, min_periods=params.noise_window).mean().shift(2)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownBrakingParams,
    ) -> SignalFrame:
        valid = indicators.notna().all(axis=1)
        entry = (
            valid
            & (data["close"] > indicators["ma200"])
            & (indicators["depth"] >= params.min_depth)
            & (indicators["prior_change"] > params.shock_mult * indicators["noise"].clip(lower=1e-8))
            & (indicators["depth_change"] <= params.braking_ratio * indicators["prior_change"])
        )
        df = pd.DataFrame(index=data.index)
        # Maintain the position only while the complete entry predicate remains true.
        # Its reversal is the sole exit mechanic; there is no timed exit.
        df["signal"] = entry.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
