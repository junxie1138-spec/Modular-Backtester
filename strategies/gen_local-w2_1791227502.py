from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    lookback: int = 20
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local-w2_1791227502"

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return 4 * params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if params.lookback < 2 or params.hold_bars < 1:
            raise ValueError("lookback must be at least 2 and hold_bars positive")
        returns = data["close"].pct_change()
        gains = returns.clip(lower=0)
        out = pd.DataFrame(index=data.index)
        out["window_return"] = returns.rolling(params.lookback, min_periods=params.lookback).sum()
        out["prior_peak"] = out["window_return"].rolling(3 * params.lookback, min_periods=3 * params.lookback).max().shift(1)
        out["participation"] = returns.gt(0).rolling(params.lookback, min_periods=params.lookback).mean()
        gain_sum = gains.rolling(params.lookback, min_periods=params.lookback).sum()
        largest_gain = gains.rolling(params.lookback, min_periods=params.lookback).max()
        out["largest_share"] = largest_gain.div(gain_sum.where(gain_sum.gt(0)))
        out["entry"] = (
            out["window_return"].gt(0)
            & out["window_return"].gt(out["prior_peak"])
            & out["participation"].ge(0.60)
            & out["largest_share"].le(0.35)
        )
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, entry in enumerate(indicators["entry"].to_numpy(dtype=bool)):
            if remaining:
                remaining -= 1
                signals[i] = int(remaining > 0)
            elif entry:
                remaining = params.hold_bars
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveParams(lookback=2, hold_bars=3)
    data = pd.DataFrame(index=range(8))
    indicators = pd.DataFrame({"entry": [True, True, True, True, False, True, True, True]})
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 1, 1]
