from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ImpactParams:
    impact_window: int = 8
    volume_window: int = 30
    volatility_window: int = 20
    volume_multiple: float = 1.15
    retention_floor: float = 0.50
    signal_threshold: float = 0.70
    target_volatility: float = 0.12
    minimum_size: float = 0.05
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[ImpactParams]):
    strategy_id = "gen_local_1790989532"

    @classmethod
    def params_type(cls) -> type[ImpactParams]:
        return ImpactParams

    def warmup_bars(self, params: ImpactParams) -> int:
        impact = max(2, int(params.impact_window))
        volume = max(2, int(params.volume_window))
        volatility = max(2, int(params.volatility_window))
        return int(max(volume + impact, volatility + 1))

    def indicators(self, data: pd.DataFrame, params: ImpactParams) -> pd.DataFrame:
        impact = max(2, int(params.impact_window))
        volume = max(2, int(params.volume_window))
        volatility = max(2, int(params.volatility_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["volume_baseline"] = data["volume"].rolling(volume, min_periods=volume).mean().shift(1)
        out["relative_volume"] = data["volume"] / out["volume_baseline"].where(out["volume_baseline"] > 0)
        out["confirmed_return"] = out["return"].where(
            out["relative_volume"] >= max(1.0, params.volume_multiple), 0.0
        ).where(out["relative_volume"].notna() & out["return"].notna())
        out["net_impact"] = out["confirmed_return"].rolling(impact, min_periods=impact).sum()
        out["gross_impact"] = out["confirmed_return"].abs().rolling(impact, min_periods=impact).sum()
        out["total_displacement"] = out["return"].rolling(impact, min_periods=impact).sum()
        out["daily_volatility"] = out["return"].rolling(volatility, min_periods=volatility).std(ddof=0)
        out["annual_volatility"] = out["daily_volatility"] * np.sqrt(252.0)
        out["coherence"] = out["net_impact"].abs() / out["gross_impact"].where(out["gross_impact"] > 0)
        out["retention"] = (
            np.sign(out["net_impact"]) * out["total_displacement"]
            / out["net_impact"].abs().where(out["net_impact"].abs() > 0)
        ).clip(0.0, 1.0)
        out["strength"] = (
            out["net_impact"].abs()
            / (out["daily_volatility"].where(out["daily_volatility"] > 0) * np.sqrt(float(impact)))
            * out["coherence"] * out["retention"]
        )
        out["ready"] = out[["coherence", "retention", "strength", "annual_volatility"]].notna().all(axis=1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ImpactParams,
    ) -> SignalFrame:
        qualifies = (
            indicators["ready"].fillna(False)
            & (indicators["coherence"] >= 0.50)
            & (indicators["retention"] >= params.retention_floor)
            & (indicators["strength"] >= params.signal_threshold)
        )
        entries = np.sign(indicators["net_impact"].fillna(0.0)).where(qualifies, 0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries.to_numpy()):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signals[i] = position
        lower = max(0.001, min(float(params.minimum_size), 1.0))
        upper = max(lower, min(float(params.maximum_size), 1.0))
        strength = indicators["strength"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        conviction = strength / (1.0 + strength)
        exposure = (
            max(0.001, float(params.target_volatility))
            / indicators["annual_volatility"].where(indicators["annual_volatility"] > 0)
            * conviction
        ).replace([np.inf, -np.inf], np.nan).fillna(lower).clip(lower, upper)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = exposure.shift(1).fillna(lower).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
