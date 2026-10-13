from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    rank_window: int = 20
    breakout_window: int = 5
    atr_window: int = 14
    rank_floor: float = 0.35
    rank_ceiling: float = 0.85
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790712235"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.rank_window + 1, params.breakout_window + 1, params.atr_window + 1)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        prev_close = data["close"].shift(1)
        true_range = pd.concat(
            [(data["high"] - data["low"]), (data["high"] - prev_close).abs(), (data["low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        normalized_range = true_range / data["close"].replace(0, np.nan)
        volatility_rank = normalized_range.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        prior_high = data["high"].shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max()
        breakout_strength = ((data["close"] - prior_high) / atr.replace(0, np.nan)).clip(lower=0.0)
        return pd.DataFrame(
            {
                "atr": atr,
                "volatility_rank": volatility_rank,
                "prior_high": prior_high,
                "breakout_strength": breakout_strength,
            },
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams
    ) -> SignalFrame:
        vol_ok = indicators["volatility_rank"].between(params.rank_floor, params.rank_ceiling)
        breakout = data["close"] > indicators["prior_high"]
        raw = (vol_ok & breakout & indicators["atr"].notna()).fillna(False)
        strength = indicators["breakout_strength"].fillna(0.0).clip(0.0, 1.0)
        size = (0.25 + 0.75 * strength).clip(0.25, 1.0)

        signal = pd.Series(0, index=data.index, dtype=int)
        entry_price = 0.0
        entry_atr = 0.0
        active = False
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            if active and close < entry_price - params.stop_atr * entry_atr:
                active = False
                entry_price = 0.0
                entry_atr = 0.0
            if not active and bool(raw.iloc[i]):
                active = True
                entry_price = close
                entry_atr = float(indicators["atr"].iloc[i])
            signal.iloc[i] = 1 if active else 0

        df = pd.DataFrame({"signal": signal, "size": size.astype(float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
