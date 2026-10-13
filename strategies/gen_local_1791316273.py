from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResidenceParams:
    lookback: int = 20
    atr_multiple: float = 2.0

    def __post_init__(self):
        if not isinstance(self.lookback, int) or isinstance(self.lookback, bool) or self.lookback < 5:
            raise ValueError('lookback must be an integer >= 5')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('atr_multiple must be finite and positive')


class GeneratedStrategy(BaseStrategy[ResidenceParams]):
    strategy_id = 'gen_local_1791316273'

    @classmethod
    def params_type(cls) -> type[ResidenceParams]:
        return ResidenceParams

    def warmup_bars(self, params: ResidenceParams) -> int:
        return max(params.lookback + 5, 15)

    def indicators(self, data: pd.DataFrame, params: ResidenceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['intraday_range'] = data['high'] - data['low']
        out['high_gap'] = (data['high'] - out['previous_close']).abs()
        out['low_gap'] = (data['low'] - out['previous_close']).abs()
        out['true_range'] = out[['intraday_range', 'high_gap', 'low_gap']].max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['close_percentile'] = data['close'].rolling(params.lookback, min_periods=params.lookback).rank(pct=True)
        out['upper_residence'] = out['close_percentile'].ge(0.8).astype(float).where(out['close_percentile'].notna())
        out['residence_fraction'] = out['upper_residence'].rolling(5, min_periods=5).mean().shift(1)
        out['shelf_high'] = data['close'].rolling(5, min_periods=5).max().shift(1)
        out['shelf_low'] = data['close'].rolling(5, min_periods=5).min().shift(1)
        out['shelf_width'] = out['shelf_high'] - out['shelf_low']
        out['previous_atr'] = out['atr'].shift(1)
        out['resistance'] = data['high'].rolling(params.lookback, min_periods=params.lookback).max().shift(1)
        out['entry'] = (
            out['residence_fraction'].ge(0.8)
            & out['shelf_width'].le(1.5 * out['previous_atr'])
            & data['close'].gt(out['resistance'])
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ResidenceParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = stop = 0.0
        entry_bar = -1
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                active = False
                continue
            if active:
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.atr_multiple * atr)
                if close <= stop or i - entry_bar >= 10:
                    active = False
            elif entries[i]:
                active = True
                entry_bar = i
                high_water = close
                stop = high_water - params.atr_multiple * atr
            signals[i] = int(active)
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = ResidenceParams()
    index = pd.date_range('2020-01-01', periods=12)
    closes = [100.0, 104.0, 103.0, 101.0] + [102.0] * 8
    data = pd.DataFrame({'open': closes, 'high': closes, 'low': closes, 'close': closes, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'atr': [1.0, 1.0] + [10.0] * 10, 'entry': [True] + [False] * 11}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1] + [0] * 8
    data.loc[:, 'close'] = 100.0
    indicators.loc[:, 'atr'] = 1.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0] + [1] * 10 + [0]
    indicators.loc[:, 'atr'] = np.nan
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
