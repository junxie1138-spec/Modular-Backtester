from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPairParams:
    lookback: int = 18
    entry_bias: float = 0.10
    min_pairs: int = 4
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapPairParams]):
    strategy_id = "gen_local-w2_1790974900"

    @classmethod
    def params_type(cls) -> type[GapPairParams]:
        return GapPairParams

    def warmup_bars(self, params: GapPairParams) -> int:
        return max(2, int(params.lookback)) + 2

    def indicators(self, data: pd.DataFrame, params: GapPairParams) -> pd.DataFrame:
        window = max(2, int(params.lookback))
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        gap = (data["open"] / prior_close - 1.0).replace([np.inf, -np.inf], np.nan)
        previous_gap = gap.shift(1)
        valid = gap.notna() & previous_gap.notna()
        opposite = (gap * previous_gap < 0) & valid
        residual = (gap + previous_gap).where(opposite, 0.0).where(valid)
        amplitude = (gap.abs() + previous_gap.abs()).where(opposite, 0.0).where(valid)
        residual_sum = residual.rolling(window, min_periods=window).sum()
        amplitude_sum = amplitude.rolling(window, min_periods=window).sum()
        pair_count = opposite.astype(float).where(valid).rolling(window, min_periods=window).sum()
        bias = (residual_sum / amplitude_sum.where(amplitude_sum > 0)).clip(-1.0, 1.0)
        return pd.DataFrame({
            "gap": gap.fillna(0.0),
            "pair_residual": residual.fillna(0.0),
            "pair_amplitude": amplitude.fillna(0.0),
            "pair_count": pair_count.fillna(0.0),
            "bias": bias.fillna(0.0),
            "ready": (pair_count.notna() & amplitude_sum.gt(0)).astype(int),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapPairParams) -> SignalFrame:
        threshold = float(np.clip(params.entry_bias, 0.0, 0.99))
        entry_condition = (
            indicators["ready"].eq(1)
            & indicators["pair_count"].ge(max(1, int(params.min_pairs)))
            & indicators["bias"].gt(threshold)
        )
        # A held position exits only when this same entry condition flips false.
        conviction = ((indicators["bias"] - threshold) / (1.0 - threshold)).clip(0.0, 1.0)
        floor = float(np.clip(params.min_size, 0.01, 1.0))
        ceiling = float(np.clip(params.max_size, floor, 1.0))
        df = pd.DataFrame(index=data.index)
        df["signal"] = entry_condition.astype(int)
        df["size"] = (floor + (ceiling - floor) * conviction).shift(1).fillna(floor).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapPairParams()
    strategy = GeneratedStrategy()
    gap = np.tile([0.012, -0.004], 40)
    gap[40:] = np.tile([0.004, -0.012], 20)
    close = 100.0 * np.cumprod(1.0 + gap)
    data = pd.DataFrame({"open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1000000.0},
                        index=pd.date_range("2020-01-01", periods=80))
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = (indicators["ready"].eq(1) & indicators["pair_count"].ge(params.min_pairs)
                & indicators["bias"].gt(params.entry_bias)).astype(int).shift(1).fillna(0).astype(int)
    assert result["signal"].equals(expected)
    assert result["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    assert result["signal"].iloc[30] == 1 and result["signal"].iloc[-1] == 0
    assert result["size"].between(params.min_size, params.max_size).all()
    assert result.loc[result["signal"].eq(1), "size"].nunique() > 1
