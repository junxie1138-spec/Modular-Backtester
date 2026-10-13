from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EquilibriumParams:
    lookback: int = 9
    entry_z: float = 1.75
    participation_ceiling: float = 0.80
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[EquilibriumParams]):
    strategy_id = "gen_local_1790988364"

    @classmethod
    def params_type(cls) -> type[EquilibriumParams]:
        return EquilibriumParams

    def warmup_bars(self, params: EquilibriumParams) -> int:
        return max(2, min(10, int(params.lookback)))

    def indicators(self, data: pd.DataFrame, params: EquilibriumParams) -> pd.DataFrame:
        window = self.warmup_bars(params)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = data["volume"].astype(float).replace([np.inf, -np.inf], np.nan)
        high = data["high"].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data["low"].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = (close > 0) & (volume >= 0) & (low > 0) & (high >= low)
        price = close.where(valid)
        weight = volume.where(valid)
        weight_sum = weight.rolling(window, min_periods=window).sum().replace(0.0, np.nan)
        equilibrium = (price * weight).rolling(window, min_periods=window).sum() / weight_sum
        second_moment = (price.pow(2) * weight).rolling(window, min_periods=window).sum() / weight_sum
        dispersion = (second_moment - equilibrium.pow(2)).clip(lower=0.0).pow(0.5)
        z_score = (price - equilibrium) / dispersion.replace(0.0, np.nan)
        relative_volume = weight / (weight_sum / window)
        log_range = np.log(high.where(valid) / low.where(valid))
        annual_vol = (log_range.pow(2).rolling(window, min_periods=window).mean() * 252.0 / (4.0 * np.log(2.0))).pow(0.5)
        ready = valid & equilibrium.notna() & dispersion.gt(0.0) & relative_volume.notna() & annual_vol.notna()
        result = pd.DataFrame(index=data.index)
        result["equilibrium"] = equilibrium.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["dispersion"] = dispersion.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["z_score"] = z_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["relative_volume"] = relative_volume.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["annual_vol"] = annual_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["ready"] = ready.fillna(False)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: EquilibriumParams) -> SignalFrame:
        threshold = max(0.1, float(params.entry_z))
        eligible = indicators["ready"] & indicators["relative_volume"].le(max(0.0, float(params.participation_ceiling)))
        desired = pd.Series(0, index=data.index, dtype=int)
        desired.loc[eligible & indicators["z_score"].le(-threshold)] = 1
        desired.loc[eligible & indicators["z_score"].ge(threshold)] = -1
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(desired.to_numpy()):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Opposite entry condition exits to flat before a new entry.
                position = 0
            positions[i] = position
        target = max(0.001, float(params.target_vol))
        allocation = (target / indicators["annual_vol"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.05).clip(lower=0.05, upper=0.95)
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["size"] = allocation.shift(1).fillna(0.05).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
