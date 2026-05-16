from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 12
    density_threshold: float = 0.60


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778907205"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # pct_change needs 1 bar, rolling(window) needs window bars, diff needs 1 more
        return params.window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        returns = data["close"].pct_change()
        # Fraction of bars with negative returns inside the rolling window
        neg_density = (returns < 0).rolling(params.window).mean()
        # First derivative of density: negative means the jam is thinning
        density_delta = neg_density.diff()

        ind = pd.DataFrame(index=data.index)
        ind["neg_density"] = neg_density
        ind["density_delta"] = density_delta
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        # Shockwave present: negative-return density above threshold (jam formed)
        in_shockwave = indicators["neg_density"] > params.density_threshold
        # Shockwave dissipating: density now declining (jam clearing)
        dissipating = indicators["density_delta"] < 0

        raw = (in_shockwave & dissipating).astype(int)
        # Signal-reversal exit: position held while both conditions remain true;
        # drops to 0 the bar either condition flips — no separate exit logic needed.
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
