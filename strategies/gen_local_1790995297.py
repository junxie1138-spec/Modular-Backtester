from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityPolarityParams:
    vol_window: int = 10
    baseline_window: int = 40
    dominance_threshold: float = 1.5
    expansion_threshold: float = 1.05
    profit_target: float = 0.015
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VolatilityPolarityParams]):
    strategy_id = "gen_local_1790995297"

    @classmethod
    def params_type(cls) -> type[VolatilityPolarityParams]:
        return VolatilityPolarityParams

    def warmup_bars(self, params: VolatilityPolarityParams) -> int:
        return int(max(200, params.vol_window + params.baseline_window + 1))

    def indicators(self, data: pd.DataFrame, params: VolatilityPolarityParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        upside = returns.clip(lower=0).pow(2).rolling(params.vol_window, min_periods=params.vol_window).mean().pow(0.5)
        downside = returns.clip(upper=0).pow(2).rolling(params.vol_window, min_periods=params.vol_window).mean().pow(0.5)
        volatility = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        baseline = volatility.shift(1).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        polarity = upside / downside.clip(lower=1e-12)
        expansion = volatility / baseline.where(baseline > 0)
        result = pd.DataFrame(index=data.index)
        result["return"] = returns
        result["upside_volatility"] = upside
        result["downside_volatility"] = downside
        result["volatility"] = volatility
        result["volatility_baseline"] = baseline
        result["polarity"] = polarity
        result["previous_polarity"] = polarity.shift(1)
        result["expansion"] = expansion
        result["regime_ma"] = close.rolling(200, min_periods=200).mean()
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VolatilityPolarityParams) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entries = (
            ready
            & (data["close"] > indicators["regime_ma"])
            & (indicators["return"] > 0)
            & (indicators["polarity"] > params.dominance_threshold)
            & (indicators["previous_polarity"] <= params.dominance_threshold)
            & (indicators["expansion"] >= params.expansion_threshold)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=np.int64)
        active = False
        pending_entry = False
        entry_price = 0.0
        entry_bar = -1
        max_hold = max(1, int(params.hold_bars))

        for i in range(len(data)):
            # A prior close's entry decision fills at this session's open.
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i

            if active:
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_expired = i - entry_bar + 1 >= max_hold
                if target_hit or time_expired:
                    active = False
                    # Zero decided at this close takes effect next session.
                    desired[i] = 0
                else:
                    desired[i] = 1
            elif entries[i] and i + 1 < len(data):
                desired[i] = 1
                pending_entry = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
