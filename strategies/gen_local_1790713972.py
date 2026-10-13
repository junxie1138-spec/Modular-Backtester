from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    pattern_window: int = 20
    trend_window: int = 20
    complexity_floor: float = 0.72
    stop_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790713972"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.pattern_window + 4, params.trend_window, 14) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()
        pattern = pd.Series(0, index=data.index, dtype="int64")
        valid = ret.notna().rolling(4).sum().eq(4)
        bits = 0
        for i in range(4):
            for j in range(i + 1, 4):
                pattern += (ret.shift(i) > ret.shift(j)).fillna(False).astype("int64") * (1 << bits)
                bits += 1

        entropy = pd.Series(0.0, index=data.index)
        for code in range(64):
            probability = pattern.eq(code).where(valid).rolling(params.pattern_window).mean()
            entropy -= probability.where(probability > 0, 1.0) * np.log(probability.where(probability > 0, 1.0))
        entropy = entropy / np.log(24.0)

        tr = pd.concat(
            [data["high"] - data["low"], (data["high"] - close.shift()).abs(), (data["low"] - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14).mean()
        momentum = close / close.shift(params.trend_window) - 1.0
        return pd.DataFrame({"entropy": entropy, "atr": atr, "momentum": momentum}, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        entropy = indicators["entropy"]
        raw = (entropy >= params.complexity_floor) & (indicators["momentum"] > 0)
        strength = ((entropy - params.complexity_floor) / max(1.0 - params.complexity_floor, 1e-9)).clip(0.0, 1.0)
        size = (0.25 + 0.75 * strength).where(raw, 1.0).astype(float)

        signal = pd.Series(0, index=data.index, dtype="int64")
        in_trade = False
        entry_price = 0.0
        entry_atr = 0.0
        for i in range(len(data)):
            if in_trade:
                if data["close"].iloc[i] < entry_price - params.stop_atr * entry_atr:
                    in_trade = False
                else:
                    signal.iloc[i] = 1
                    continue
            if raw.iloc[i] and pd.notna(indicators["atr"].iloc[i]):
                in_trade = True
                entry_price = float(data["close"].iloc[i])
                entry_atr = float(indicators["atr"].iloc[i])
                signal.iloc[i] = 1

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
