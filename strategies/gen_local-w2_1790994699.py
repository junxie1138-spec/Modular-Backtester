from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    streak_length: int = 2
    vol_window: int = 20
    target_vol: float = 0.12
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790994699"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.vol_window + 1, params.streak_length + 3))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = returns.shift(1)
        decaying_loss = (returns < 0) & (previous < 0) & (returns > previous)
        groups = (~decaying_loss).cumsum()
        streak = decaying_loss.astype(int).groupby(groups).cumsum().clip(upper=params.streak_length)
        out = pd.DataFrame(index=data.index)
        out["returns"] = returns
        out["previous_return"] = previous
        out["decaying_loss"] = decaying_loss
        out["streak"] = streak
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["annual_vol"] = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["trigger"] = (streak == params.streak_length) & (streak.shift(1).fillna(0) < params.streak_length)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if params.streak_length < 1 or params.vol_window < 2:
            raise ValueError("streak_length must be positive and vol_window at least two")
        if params.holding_bars not in (1, 2) or not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("holding_bars must be one or two and target_vol positive")
        vol = indicators["annual_vol"]
        eligible = (
            indicators["trigger"].fillna(False)
            & (data["close"] > indicators["ma200"])
            & vol.notna()
            & np.isfinite(vol)
            & (vol > 0)
        ).to_numpy(dtype=bool)
        exposure = (params.target_vol / vol.clip(lower=1e-8)).clip(lower=0.01, upper=0.95).fillna(0.01).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.01, dtype=float)
        exit_bar = -1
        entry_size = 0.01
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = entry_size
            elif eligible[i]:
                exit_bar = i + params.holding_bars
                entry_size = float(exposure[i])
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
