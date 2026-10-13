from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityTakeoverParams:
    recent_window: int = 5
    baseline_window: int = 20
    expansion_threshold: float = 1.15
    takeover_threshold: float = 0.70
    prior_share_ceiling: float = 0.55
    profit_target: float = 0.06
    max_hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[VolatilityTakeoverParams]):
    strategy_id = "gen_local_1790992927"

    @classmethod
    def params_type(cls) -> type[VolatilityTakeoverParams]:
        return VolatilityTakeoverParams

    @staticmethod
    def warmup_bars(params: VolatilityTakeoverParams) -> int:
        return int(max(2, params.recent_window) + max(2, params.baseline_window) + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolatilityTakeoverParams) -> pd.DataFrame:
        recent = max(2, int(params.recent_window))
        baseline = max(2, int(params.baseline_window))
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        upside = returns.clip(lower=0.0)
        downside = returns.clip(upper=0.0)
        recent_up = upside.rolling(recent, min_periods=recent).std(ddof=0)
        recent_down = downside.rolling(recent, min_periods=recent).std(ddof=0)
        baseline_up = upside.shift(recent).rolling(baseline, min_periods=baseline).std(ddof=0)
        baseline_down = downside.shift(recent).rolling(baseline, min_periods=baseline).std(ddof=0)
        recent_energy = recent_up.pow(2) + recent_down.pow(2)
        baseline_energy = baseline_up.pow(2) + baseline_down.pow(2)
        out = pd.DataFrame(index=data.index)
        out["returns"] = returns
        out["recent_up_std"] = recent_up
        out["recent_down_std"] = recent_down
        out["baseline_up_std"] = baseline_up
        out["baseline_down_std"] = baseline_down
        out["recent_volatility"] = returns.rolling(recent, min_periods=recent).std(ddof=0)
        out["baseline_volatility"] = returns.shift(recent).rolling(baseline, min_periods=baseline).std(ddof=0)
        out["upside_share"] = recent_up.pow(2).div(recent_energy.where(recent_energy > 0.0))
        out["prior_upside_share"] = baseline_up.pow(2).div(baseline_energy.where(baseline_energy > 0.0))
        out["volatility_expansion"] = out["recent_volatility"].div(out["baseline_volatility"].where(out["baseline_volatility"] > 0.0))
        out["recent_drift"] = returns.rolling(recent, min_periods=recent).sum()
        out["takeover_strength"] = (
            (out["upside_share"] - out["prior_upside_share"]).clip(lower=0.0)
            * (out["volatility_expansion"] - 1.0).clip(lower=0.0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityTakeoverParams,
    ) -> SignalFrame:
        qualified = (
            indicators["volatility_expansion"].ge(params.expansion_threshold)
            & indicators["upside_share"].ge(params.takeover_threshold)
            & indicators["prior_upside_share"].le(params.prior_share_ceiling)
            & indicators["recent_drift"].gt(0.0)
        ).fillna(False)
        triggers = (qualified & ~qualified.shift(1, fill_value=False)).to_numpy(dtype=bool)
        strength = indicators["takeover_strength"].fillna(0.0).clip(lower=0.0)
        conviction = strength.div(strength + 0.10).to_numpy(dtype=float)
        min_size = float(np.clip(params.min_size, 0.01, 1.0))
        max_size = float(np.clip(params.max_size, min_size, 1.0))
        hold_limit = max(1, int(params.max_hold_bars))
        target = max(0.001, float(params.profit_target))
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), min_size, dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        position_size = min_size

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                profit_exit = valid_entry and np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + target)
                time_exit = i - entry_bar + 1 >= hold_limit
                if not valid_entry or profit_exit or time_exit:
                    active = False
                else:
                    signals[i] = 1
                sizes[i] = position_size
            elif triggers[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                position_size = min_size + (max_size - min_size) * conviction[i]
                signals[i] = 1
                sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
