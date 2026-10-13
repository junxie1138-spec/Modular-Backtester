from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    window: int = 18
    lag: int = 2
    threshold: float = 0.02


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local-w2_1791253925'

    @classmethod
    def params_type(cls) -> type[OrdinalParams]:
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return params.window + 2 * params.lag

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        if params.window < 2 or params.lag < 1 or not 0.0 < params.threshold < 1.0:
            raise ValueError('Require window >= 2, lag >= 1, and 0 < threshold < 1.')
        close = data['close'].astype(float)
        triples = pd.concat([close.shift(2 * params.lag), close.shift(params.lag), close], axis=1)
        valid = np.isfinite(triples).all(axis=1) & triples.gt(0).all(axis=1)
        # Stable sorting resolves equal prices by chronological order.
        order = np.argsort(triples.to_numpy(), axis=1, kind='stable')
        out = pd.DataFrame(index=data.index)
        out['pattern'] = pd.Series(order @ np.array([9, 3, 1]), index=data.index).where(valid)
        columns = []
        for code in (5, 7, 11, 15, 19, 21):
            name = f'p_{code}'
            columns.append(name)
            out[name] = out['pattern'].eq(code).where(valid).rolling(params.window, min_periods=params.window).mean()
        probabilities = out[columns]
        log_probabilities = np.log(probabilities.where(probabilities > 0, 1.0))
        out['entropy'] = -(probabilities * log_probabilities).sum(axis=1, min_count=6) / np.log(6.0)
        out['dominance'] = out['p_5'] - out['p_21']
        out['score'] = out['dominance'] * (1.0 - out['entropy']).clip(0.0, 1.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        score = indicators['score']
        # Mirrored thresholds: bullish dominance enters; bearish dominance exits.
        # Neutral or unavailable evidence preserves the previous position.
        decisions = pd.Series(np.select([score > params.threshold, score < -params.threshold], [1.0, 0.0], default=np.nan), index=data.index)
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions.ffill().fillna(0).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = OrdinalParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=160)
    close = np.concatenate([np.arange(100.0, 180.0), np.arange(178.0, 98.0, -1.0)])
    data = pd.DataFrame({'open': close, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000000.0}, index=index)
    features = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, features, None, params).data
    assert signals['signal'].iloc[:strategy.warmup_bars(params)].eq(0).all()
    assert signals['signal'].iloc[50] == 1
    assert signals['signal'].iloc[-1] == 0
    assert signals['signal'].isin([0, 1]).all()
    assert signals['size'].gt(0).all()
    prefix = data.iloc[:100]
    prefix_signals = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:100], prefix_signals)
