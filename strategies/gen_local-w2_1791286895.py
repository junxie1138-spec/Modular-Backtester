from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResetParams:
    volatility_window: int = 18
    momentum_window: int = 18
    regime_margin: float = 0.10


class GeneratedStrategy(BaseStrategy[ResetParams]):
    strategy_id = "gen_local-w2_1791286895"

    @classmethod
    def params_type(cls) -> type[ResetParams]:
        return ResetParams

    @staticmethod
    def warmup_bars(params: ResetParams) -> int:
        return int(max(2 * params.volatility_window + 1, params.momentum_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ResetParams) -> pd.DataFrame:
        if params.volatility_window < 2 or params.momentum_window < 1:
            raise ValueError("Lookbacks must support volatility and momentum calculations")
        if not 0.0 < params.regime_margin < 1.0:
            raise ValueError("regime_margin must be between zero and one")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        window = params.volatility_window
        out["volatility"] = out["return"].rolling(window, min_periods=window).std(ddof=0)
        out["previous_volatility"] = out["volatility"].shift(window)
        out["volatility_ratio"] = out["volatility"].div(out["previous_volatility"].where(out["previous_volatility"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["momentum"] = close.pct_change(params.momentum_window, fill_method=None).replace([np.inf, -np.inf], np.nan)
        months = pd.Series(data.index.to_period("M"), index=data.index)
        out["monthly_reset"] = months.ne(months.shift(1))
        if len(out):
            out.iloc[0, out.columns.get_loc("monthly_reset")] = False
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ResetParams) -> SignalFrame:
        valid = indicators[["volatility_ratio", "momentum"]].notna().all(axis=1)
        reset = indicators["monthly_reset"] & valid
        long_entry = reset & (indicators["volatility_ratio"] < 1.0 - params.regime_margin) & (indicators["momentum"] > 0.0)
        short_entry = reset & (indicators["volatility_ratio"] > 1.0 + params.regime_margin) & (indicators["momentum"] < 0.0)
        decisions = pd.Series(np.nan, index=data.index, dtype=float)
        decisions.loc[long_entry] = 1.0
        decisions.loc[short_entry] = -1.0
        desired = decisions.ffill().fillna(0.0).astype(int).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, target in enumerate(desired):
            if position != 0 and target == -position:
                position = 0
            elif position == 0 and target != 0:
                position = int(target)
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
