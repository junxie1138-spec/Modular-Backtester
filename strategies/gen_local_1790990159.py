from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    min_streak: int = 2
    contraction_ratio: float = 0.90
    expansion_ratio: float = 1.10


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1790990159"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return max(200, params.min_streak + 1)

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["previous_close"] = data["close"].shift(1)
        denominator = out["previous_close"].where(out["previous_close"].gt(0))
        out["body"] = ((data["close"] - data["open"]) / denominator).replace([np.inf, -np.inf], np.nan)
        out["candle_direction"] = np.sign(out["body"]).fillna(0).astype(int)
        out["streak_group"] = out["candle_direction"].ne(out["candle_direction"].shift(1)).cumsum()
        out["streak"] = out["candle_direction"].ne(0).astype(int).groupby(out["streak_group"]).cumsum()
        out["speed"] = out["body"].abs()
        out["previous_speed"] = out["speed"].shift(1)
        out["bull_regime"] = data["close"].ge(out["ma200"])
        out["ready"] = out["ma200"].notna() & out["body"].notna() & out["previous_speed"].notna()
        out["setup_direction"] = np.where(out["bull_regime"], -out["candle_direction"], out["candle_direction"])
        out["contracting"] = out["speed"].le(out["previous_speed"] * params.contraction_ratio)
        out["expanding"] = out["speed"].ge(out["previous_speed"] * params.expansion_ratio)
        out["entry"] = out["ready"] & out["streak"].ge(params.min_streak) & ((out["bull_regime"] & out["contracting"]) | (~out["bull_regime"] & out["expanding"]))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        position = 0
        signals = np.zeros(len(data), dtype=int)
        rows = zip(indicators["ready"], indicators["setup_direction"], indicators["entry"])
        for i, (ready, direction, entry) in enumerate(rows):
            if ready:
                if position != 0:
                    # Exit only when the regime-mapped directional setup flips.
                    # Neutral candles and lost entry qualification preserve the position.
                    if direction == -position:
                        position = 0
                elif entry and direction != 0:
                    position = int(direction)
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
