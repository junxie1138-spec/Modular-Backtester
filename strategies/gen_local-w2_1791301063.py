from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationBreakoutParams:
    slope_span: int = 8
    breakout_bars: int = 20
    rank_bars: int = 126
    innovation_q: float = 0.85
    atr_bars: int = 14
    breakeven_pct: float = 0.03
    atr_mult: float = 2.5
    max_hold: int = 10
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[InnovationBreakoutParams]):
    strategy_id = "gen_local-w2_1791301063"

    @classmethod
    def params_type(cls) -> type[InnovationBreakoutParams]:
        return InnovationBreakoutParams

    def warmup_bars(self, params: InnovationBreakoutParams) -> int:
        return max(params.rank_bars + params.slope_span + 2, params.breakout_bars + 1, params.atr_bars + 1)

    def indicators(self, data: pd.DataFrame, params: InnovationBreakoutParams) -> pd.DataFrame:
        log_return = np.log(data["close"]).diff()
        slope = log_return.ewm(span=params.slope_span, adjust=False, min_periods=params.slope_span).mean()
        innovation = log_return - slope.shift(1)
        threshold = innovation.shift(1).rolling(params.rank_bars, min_periods=params.rank_bars).quantile(params.innovation_q)
        prior_high = data["high"].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).max()
        previous_close = data["close"].shift(1)
        true_range = pd.concat((data["high"] - data["low"], (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()), axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        ready = threshold.notna() & prior_high.notna() & atr.notna()
        return pd.DataFrame({
            "slope": slope.fillna(0.0),
            "innovation": innovation.fillna(0.0),
            "threshold": threshold.fillna(0.0),
            "prior_high": prior_high.fillna(0.0),
            "atr": atr.fillna(0.0),
            "ready": ready,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: InnovationBreakoutParams) -> SignalFrame:
        entry_condition = (indicators["ready"] & (indicators["slope"] > 0) & (indicators["innovation"] > indicators["threshold"]) & (data["close"] > indicators["prior_high"]))
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = float(params.size)
        active = False
        armed = False
        held = 0
        entry = 0.0
        stop = 0.0
        for i in range(len(data)):
            close = float(data["close"].iat[i])
            atr = float(indicators["atr"].iat[i])
            if active:
                if held == 0:
                    entry = float(data["open"].iat[i])
                    stop = entry - params.atr_mult * atr
                held += 1
                if close <= stop or held >= params.max_hold:
                    active = False
                else:
                    if close >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, close - params.atr_mult * atr)
                    df.iat[i, 0] = 1
            elif bool(entry_condition.iat[i]):
                active = True
                armed = False
                held = 0
                df.iat[i, 0] = 1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
