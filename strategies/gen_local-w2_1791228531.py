from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTensionParams:
    tension_window: int = 18
    reversal_threshold: float = 0.12
    exposure: float = 0.95

    def __post_init__(self):
        if self.tension_window < 2:
            raise ValueError('tension_window must be at least 2')
        if not np.isfinite(self.reversal_threshold) or self.reversal_threshold <= 0:
            raise ValueError('reversal_threshold must be positive and finite')
        if not np.isfinite(self.exposure) or not 0 < self.exposure <= 1:
            raise ValueError('exposure must be in (0, 1]')


class GeneratedStrategy(BaseStrategy[RangeTensionParams]):
    strategy_id = 'gen_local-w2_1791228531'

    @classmethod
    def params_type(cls) -> type[RangeTensionParams]:
        return RangeTensionParams

    def warmup_bars(self, params: RangeTensionParams) -> int:
        return int(max(200, params.tension_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeTensionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['range'] = (data['high'] - data['low']).where(data['high'] >= data['low'])
        out['lower_extension'] = (-data['low'].diff()).clip(lower=0)
        out['upper_extension'] = data['high'].diff().clip(lower=0)
        out['boundary_imbalance'] = out['lower_extension'] - out['upper_extension']
        out['mean_imbalance'] = out['boundary_imbalance'].rolling(
            params.tension_window, min_periods=params.tension_window
        ).mean()
        out['mean_range'] = out['range'].rolling(
            params.tension_window, min_periods=params.tension_window
        ).mean()
        out['tension'] = (
            out['mean_imbalance'] / out['mean_range'].where(out['mean_range'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeTensionParams,
    ) -> SignalFrame:
        tension = indicators['tension']
        entry = (
            tension.ge(params.reversal_threshold)
            & data['close'].gt(indicators['ma200'])
        ).fillna(False)
        reversal = tension.le(-params.reversal_threshold).fillna(False)
        events = pd.Series(np.nan, index=data.index, dtype=float)
        events.loc[entry] = 1.0
        events.loc[reversal] = 0.0
        df = pd.DataFrame(index=data.index)
        df['signal'] = events.ffill().fillna(0).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.exposure)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = RangeTensionParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'close': [101, 101, 99, 99, 99, 99, 101]}, index=index)
    indicators = pd.DataFrame(
        {'tension': [np.nan, 0.2, 0.05, 0.2, -0.2, 0.2, 0.2], 'ma200': 100.0},
        index=index,
    )
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert result['size'].gt(0).all()
    assert result.index.equals(index)
    assert strategy.warmup_bars(params) >= 200
    long_params = RangeTensionParams(tension_window=220)
    assert strategy.warmup_bars(long_params) == 221
    flat = pd.DataFrame(
        {'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1000.0},
        index=pd.date_range('2020-01-01', periods=240),
    )
    flat_indicators = strategy.indicators(flat, params)
    assert flat_indicators['tension'].isna().all()
    assert strategy.generate_signals(flat, flat_indicators, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
