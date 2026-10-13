from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WickErosionParams:
    min_streak: int = 2
    vol_window: int = 20
    min_body_fraction: float = 0.45
    target_daily_vol: float = 0.008


class GeneratedStrategy(BaseStrategy[WickErosionParams]):
    strategy_id = "gen_local-w2_1790990300"

    @classmethod
    def params_type(cls) -> type[WickErosionParams]:
        return WickErosionParams

    def warmup_bars(self, params: WickErosionParams) -> int:
        return max(max(2, int(params.vol_window)) + 1,
                   max(1, int(params.min_streak)) + 2)

    def indicators(self, data: pd.DataFrame, params: WickErosionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        body = data["close"] - data["open"]
        body_fraction = (body.abs() / span).clip(0.0, 1.0)
        wick_share = 1.0 - body_fraction
        qualifying = (
            (body > 0)
            & (body_fraction >= float(params.min_body_fraction))
            & (wick_share < wick_share.shift(1))
            & span.notna()
            & span.shift(1).notna()
        ).fillna(False)
        streak = qualifying.astype(int).groupby((~qualifying).cumsum()).cumsum()
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        window = max(2, int(params.vol_window))
        volatility = returns.rolling(window, min_periods=window).std(ddof=0)
        ready = volatility.notna() & span.notna()
        strength = (streak / (streak + 2.0)) * body_fraction.fillna(0.0)
        out["range"] = span.fillna(0.0)
        out["body"] = body.fillna(0.0)
        out["body_fraction"] = body_fraction.fillna(0.0)
        out["wick_share"] = wick_share.fillna(1.0)
        out["qualifying"] = qualifying.astype(int)
        out["streak"] = streak.astype(int)
        out["return"] = returns.fillna(0.0)
        out["daily_volatility"] = volatility.fillna(0.0)
        out["ready"] = ready.astype(int)
        out["strength"] = strength.fillna(0.0).clip(0.0, 1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WickErosionParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        entry_condition = (
            (indicators["streak"] >= max(1, int(params.min_streak)))
            & (indicators["ready"] == 1)
        )
        # Maintain exposure while the entry condition holds; its reversal exits.
        df["signal"] = entry_condition.astype(int)
        volatility_budget = (
            max(0.000001, float(params.target_daily_vol))
            / indicators["daily_volatility"].clip(lower=0.0001)
        )
        decision_size = (volatility_budget * indicators["strength"]).clip(
            lower=0.000001, upper=0.95
        )
        # Execute both the decision and its associated sizing next session.
        df["size"] = decision_size.shift(1).fillna(0.000001).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
