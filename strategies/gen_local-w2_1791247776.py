from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class InnovationConsensusParams:
    level_span: int = 10
    slope_span: int = 5
    consensus_window: int = 10
    consensus_threshold: float = 0.7
    volatility_window: int = 20
    atr_window: int = 14
    atr_multiple: float = 3.0
    target_volatility: float = 0.12
    max_hold: int = 18

    def __post_init__(self):
        if min(self.level_span, self.slope_span, self.consensus_window,
               self.volatility_window, self.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if not 0.5 < self.consensus_threshold <= 1.0:
            raise ValueError('Consensus threshold must exceed 0.5 and not exceed 1.')
        if not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError('ATR multiple must be finite and positive.')
        if not np.isfinite(self.target_volatility) or self.target_volatility <= 0:
            raise ValueError('Target volatility must be finite and positive.')
        if self.max_hold < 1:
            raise ValueError('Holding period must be positive.')


class GeneratedStrategy(BaseStrategy[InnovationConsensusParams]):
    strategy_id = 'gen_local-w2_1791247776'

    @classmethod
    def params_type(cls):
        return InnovationConsensusParams

    def warmup_bars(self, params: InnovationConsensusParams) -> int:
        return int(max(200,
                       params.level_span + params.slope_span + params.consensus_window + 2,
                       params.volatility_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame,
                   params: InnovationConsensusParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['log_close'] = np.log(close)
        out['level'] = out['log_close'].ewm(
            span=params.level_span, adjust=False,
            min_periods=params.level_span).mean()
        out['slope'] = out['level'].diff().ewm(
            span=params.slope_span, adjust=False,
            min_periods=params.slope_span).mean()
        out['innovation'] = out['slope'].diff()
        positive = out['innovation'].gt(0).astype(float).where(
            out['innovation'].notna())
        out['consensus'] = positive.rolling(
            params.consensus_window, min_periods=params.consensus_window).mean()
        out['innovation_sum'] = out['innovation'].rolling(
            params.consensus_window, min_periods=params.consensus_window).sum()
        out['returns'] = out['log_close'].diff()
        out['volatility'] = out['returns'].rolling(
            params.volatility_window,
            min_periods=params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['target_size'] = (params.target_volatility /
                              out['volatility'].clip(lower=0.01)).clip(0.05, 0.95)
        ready = np.isfinite(out[['slope', 'consensus', 'innovation_sum',
                                'volatility', 'atr', 'ma200']]).all(axis=1)
        consensus = (ready & close.gt(out['ma200']) & out['atr'].gt(0)
                     & out['slope'].gt(0) & out['innovation_sum'].gt(0)
                     & out['consensus'].ge(params.consensus_threshold))
        out['entry'] = consensus & ~consensus.shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: InnovationConsensusParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = 0
        high_water = stop = -np.inf
        values = pd.DataFrame({
            'close': data['close'],
            'atr': indicators['atr'],
            'ma200': indicators['ma200'],
            'target_size': indicators['target_size'],
            'entry': indicators['entry'].fillna(False)
        }, index=data.index)
        for i, row in enumerate(values.itertuples(index=False)):
            price, atr, ma200, size, entry = row
            valid = (np.isfinite([price, atr, ma200, size]).all()
                     and price > 0 and atr > 0 and size > 0)
            if not valid:
                active = False
                continue
            if active:
                # Entry decided at the previous close fills on this bar.
                # Only closes from actual holding bars enter the high-water mark.
                high_water = max(high_water, price)
                stop = max(stop, high_water - params.atr_multiple * atr)
                if (price < stop or price <= ma200
                        or i - entry_bar + 1 >= params.max_hold):
                    active = False
            elif entry and price > ma200:
                active = True
                entry_bar = i + 1
                high_water = stop = -np.inf
            raw[i] = int(active)
        df = pd.DataFrame({'signal': raw}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        sizing = indicators['target_size'].replace([np.inf, -np.inf], np.nan)
        df['size'] = sizing.shift(1).fillna(0.05).clip(0.05, 0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8, freq='B')
    data = pd.DataFrame({'close': [100, 104, 103, 102, 101, 100, 101, 102]},
                        index=index)
    indicators = pd.DataFrame({
        'atr': [1, 1, 5, 5, 5, 1, 1, 1],
        'ma200': 90.0,
        'target_size': 0.4,
        'entry': [True, False, False, False, False, True, False, False]
    }, index=index)
    strategy = GeneratedStrategy()
    params = InnovationConsensusParams(atr_multiple=2.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 1, 1]
    assert result['size'].gt(0).all()
    params.max_hold = 3
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0, 1, 1]
    params.max_hold = 18
    gated = indicators.copy()
    gated.loc[index[2], 'ma200'] = 200.0
    result = strategy.generate_signals(data, gated, None, params).data
    assert result['signal'].iloc[:4].tolist() == [0, 1, 1, 0]
    full_index = pd.date_range('2021-01-01', periods=250, freq='B')
    close = 100.0 + np.arange(250) * 0.1 + np.sin(np.arange(250) / 5.0)
    full = pd.DataFrame({'open': close, 'high': close + 1,
                         'low': close - 1, 'close': close,
                         'volume': 1000000.0}, index=full_index)
    derived = strategy.indicators(full, params)
    prefix = strategy.indicators(full.iloc[:220], params)
    pd.testing.assert_frame_equal(derived.iloc[:220], prefix)
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
