from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_share: float = 0.65
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791304404"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 24

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = returns.shift(1)
        older = returns.shift(2)
        oldest = returns.shift(3)
        fine = (returns - previous) / np.sqrt(2.0)
        older_fine = (older - oldest) / np.sqrt(2.0)
        medium = (returns + previous - older - oldest) / 2.0
        approximation = (returns + previous + older + oldest) / 2.0
        coarse_energy = approximation.pow(2)
        total_energy = coarse_energy + medium.pow(2) + fine.pow(2) + older_fine.pow(2)
        share = coarse_energy.div(total_energy.where(total_energy > 0))
        volatility = np.sqrt(total_energy.rolling(20, min_periods=20).mean() * 252.0 / 4.0)
        return pd.DataFrame({
            "return": returns,
            "fine": fine,
            "older_fine": older_fine,
            "medium": medium,
            "approximation": approximation,
            "coarse_energy": coarse_energy,
            "total_energy": total_energy,
            "energy_share": share,
            "annual_volatility": volatility,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        threshold = float(np.clip(params.energy_share, 0.0, 1.0))
        entry_condition = (
            indicators["return"].lt(0.0)
            & indicators["approximation"].gt(0.0)
            & indicators["energy_share"].ge(threshold)
            & indicators["annual_volatility"].notna()
        ).fillna(False)
        # Maintain exposure while the entry condition holds; its reversal exits.
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        volatility = indicators["annual_volatility"].clip(lower=1e-6)
        target = max(float(params.target_vol), 1e-6)
        df["size"] = (target / volatility).replace([np.inf, -np.inf], np.nan).clip(lower=0.01, upper=0.95)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
