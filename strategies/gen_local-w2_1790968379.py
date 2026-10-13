from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    window: int = 4
    area_threshold: float = 0.08

    def __post_init__(self):
        if self.window < 2:
            raise ValueError('window must be at least 2')
        if not 0.0 <= self.area_threshold < 0.5:
            raise ValueError('area_threshold must be in [0, 0.5)')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1790968379'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(params.window + 2)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        n = params.window
        out = pd.DataFrame(index=data.index)
        opening = data['open'].where(data['open'] > 0)
        closing = data['close'].where(data['close'] > 0)
        out['gap'] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out['body'] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        g, b = out['gap'], out['body']
        area = pd.Series(0.0, index=data.index)
        # Second-order signed area: half the sum of ordered increment wedges.
        for lag in range(1, n):
            wedge = g.shift(lag) * b - b.shift(lag) * g
            area = area + 0.5 * wedge.rolling(n - lag, min_periods=n - lag).sum()
        out['area'] = area
        out['gap_mass'] = g.abs().rolling(n, min_periods=n).sum()
        out['body_mass'] = b.abs().rolling(n, min_periods=n).sum()
        out['gap_total'] = g.rolling(n, min_periods=n).sum()
        out['body_total'] = b.rolling(n, min_periods=n).sum()
        denominator = out['gap_mass'] * out['body_mass']
        out['repair_score'] = (-area / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        valid = np.isfinite(indicators[['repair_score', 'gap_total', 'body_total']]).all(axis=1)
        condition = (
            valid
            & indicators['repair_score'].gt(params.area_threshold)
            & indicators['gap_total'].lt(0.0)
            & indicators['body_total'].gt(0.0)
        ).fillna(False)
        # Two true closes enter; the first false close exits. No timed exits.
        confirmed = condition & condition.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df['signal'] = confirmed.astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = SignatureParams(window=2)
    index = pd.date_range('2020-01-01', periods=3)
    data = pd.DataFrame({
        'open': [100.0, 90.0, 90.0],
        'high': [100.0, 90.0, 99.0],
        'low': [100.0, 90.0, 90.0],
        'close': [100.0, 90.0, 99.0],
        'volume': [1000.0, 1000.0, 1000.0],
    }, index=index)
    indicators = GeneratedStrategy.indicators(None, data, params)
    assert np.isclose(indicators['repair_score'].iloc[-1], 0.5)
    index = pd.date_range('2020-02-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'repair_score': [np.nan, 0.2, 0.2, 0.2, -0.2, 0.2, 0.2, -0.2],
        'gap_total': -0.01,
        'body_total': 0.01,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert result.data['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
