from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    recoil_window: int = 4
    rank_window: int = 5
    rank_threshold: float = 0.6
    target_vol: float = 0.12

    def __post_init__(self):
        if not (2 <= self.recoil_window and 2 <= self.rank_window and self.recoil_window + self.rank_window <= 10):
            raise ValueError('Windows must be >= 2 and sum to <= 10.')
        if not (0 < self.rank_threshold <= 1 and np.isfinite(self.target_vol) and self.target_vol > 0):
            raise ValueError('Invalid percentile threshold or volatility target.')


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = 'gen_local-w2_1791318771'

    @classmethod
    def params_type(cls) -> type[RecoilParams]:
        return RecoilParams

    @staticmethod
    def warmup_bars(params: RecoilParams) -> int:
        return int(params.recoil_window + params.rank_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['downside'] = (out['previous_close'] - data['low']).clip(lower=0)
        out['damage'] = (out['previous_close'] - data['close']).clip(lower=0)
        out['recovered'] = (out['downside'] - out['damage']).clip(lower=0)
        out['downside_sum'] = out['downside'].rolling(params.recoil_window).sum()
        out['recovered_sum'] = out['recovered'].rolling(params.recoil_window).sum()
        out['recovery_fraction'] = out['recovered_sum'].div(out['downside_sum'].replace(0, np.nan)).clip(0, 1)
        out['recovery_rank'] = out['recovery_fraction'].rolling(params.rank_window).rank(method='average', pct=True)
        out['return'] = data['close'].div(out['previous_close'].replace(0, np.nan)).sub(1)
        out['annual_vol'] = out['return'].rolling(params.recoil_window).std(ddof=0) * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoilParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        entry = (
            (indicators['recovery_rank'] >= params.rank_threshold)
            & (indicators['recovery_fraction'] > 0.5)
            & (indicators['annual_vol'] > 0)
        )
        # Hold while the entry condition remains true; exit only on its reversal.
        df['signal'] = entry.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        exposure = (params.target_vol / indicators['annual_vol'].clip(lower=0.01)).clip(0.05, 0.95)
        df['size'] = exposure.shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = RecoilParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'recovery_rank': [0.8, 0.8, 0.2, 0.8, 0.8],
        'recovery_fraction': 0.8,
        'annual_vol': 0.2,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1]
    assert result.index.equals(index) and (result['size'] > 0).all()
    assert np.allclose(result['size'].iloc[1:], 0.6)
    assert all(GeneratedStrategy.warmup_bars(RecoilParams(recoil_window=w)) <= 10 for w in (3, 4, 5))
    index = pd.date_range('2020-01-01', periods=30)
    close = pd.Series(100 + np.arange(30) * 0.1 + np.sin(np.arange(30)), index=index)
    data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000}, index=index)
    full = GeneratedStrategy.indicators(data, params)
    assert np.isfinite(full.to_numpy()).all()
    for length in (10, 20):
        pd.testing.assert_frame_equal(GeneratedStrategy.indicators(data.iloc[:length], params), full.iloc[:length])


if __name__ == '__main__':
    _self_check()
