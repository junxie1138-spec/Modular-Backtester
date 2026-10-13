from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    lookback: int = 10
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = "gen_local_1791249163"

    @classmethod
    def params_type(cls) -> type[RecoilParams]:
        return RecoilParams

    @staticmethod
    def warmup_bars(params: RecoilParams) -> int:
        if params.lookback < 6:
            raise ValueError("lookback must be at least 6")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")
        return max(int(params.lookback), 6)

    def indicators(self, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        valid = (data['high'] >= data['low']) & (data['low'] > 0)
        out['range'] = (data['high'] - data['low']).where(valid)
        out['midpoint'] = ((data['high'] + data['low']) / 2.0).where(valid)
        out['weight'] = 1.0 / out['range'].replace(0.0, np.nan)
        out['weighted_midpoint'] = out['midpoint'] * out['weight']
        out['weight_sum'] = out['weight'].rolling(n, min_periods=n).sum()
        out['weighted_midpoint_sum'] = out['weighted_midpoint'].rolling(n, min_periods=n).sum()
        out['equilibrium'] = out['weighted_midpoint_sum'] / out['weight_sum'].replace(0.0, np.nan)
        out['mean_range'] = out['range'].rolling(n, min_periods=n).mean()
        out['fast_range'] = out['range'].rolling(3, min_periods=3).mean()
        out['prior_fast_range'] = out['fast_range'].shift(3)
        out['strain'] = (data['close'] - out['equilibrium']) / out['mean_range'].replace(0.0, np.nan)
        out['contracting'] = ((out['fast_range'] < out['mean_range']) & (out['fast_range'] < out['prior_fast_range'])).astype(bool)
        out['log_range'] = np.log(data['high'].where(valid) / data['low'].where(valid))
        out['range_variance'] = out['log_range'].pow(2) / (4.0 * np.log(2.0))
        out['annual_vol'] = np.sqrt(out['range_variance'].rolling(n, min_periods=n).mean() * 252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoilParams) -> SignalFrame:
        valid = indicators[['strain', 'annual_vol']].notna().all(axis=1)
        active = indicators['contracting'].fillna(False) & valid
        candidates = np.select(
            [active & (indicators['strain'] <= -0.75), active & (indicators['strain'] >= 0.75)],
            [1, -1], default=0,
        ).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        allocation = (params.target_vol / indicators['annual_vol'].clip(lower=0.01)).clip(lower=0.05, upper=0.95)
        df['size'] = allocation.shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'strain': [-1.0, -1.0, 0.0, 1.0, 1.0, 1.0], 'contracting': [True] * 6, 'annual_vol': [0.24] * 6}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, RecoilParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1]
    assert np.allclose(result['size'].to_numpy(), [0.05, 0.5, 0.5, 0.5, 0.5, 0.5])
