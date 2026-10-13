from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityEchoParams:
    volatility_window: int = 5
    correlation_window: int = 20
    percentile_window: int = 126
    volatility_percentile: float = 0.80
    hold_bars: int = 2

    def __post_init__(self):
        if self.volatility_window < 2 or self.correlation_window < 3:
            raise ValueError('Volatility and correlation windows are too short.')
        if self.percentile_window < 2:
            raise ValueError('Percentile window must be at least two.')
        if not 0.0 < self.volatility_percentile < 1.0:
            raise ValueError('Volatility percentile must be between zero and one.')
        if self.hold_bars not in (1, 2):
            raise ValueError('Holding period must be one or two bars.')


class GeneratedStrategy(BaseStrategy[VolatilityEchoParams]):
    strategy_id = 'gen_local-w2_1791232725'

    @classmethod
    def params_type(cls):
        return VolatilityEchoParams

    @staticmethod
    def warmup_bars(params):
        return int(params.volatility_window + max(params.percentile_window + 2, params.correlation_window + 2) + 1)

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volatility'] = out['return'].rolling(params.volatility_window, min_periods=params.volatility_window).std()
        out['volatility_change'] = out['volatility'].diff()
        out['lagged_volatility_change'] = out['volatility_change'].shift(1)
        out['autocorrelation'] = out['volatility_change'].rolling(params.correlation_window, min_periods=params.correlation_window).corr(out['lagged_volatility_change']).replace([np.inf, -np.inf], np.nan)
        out['threshold'] = out['volatility'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.volatility_percentile)
        out['eruption'] = (out['volatility'] > out['threshold']) & (out['volatility'].shift(1) <= out['threshold'].shift(1))
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx: StrategyContext, params):
        required = indicators[['return', 'volatility', 'threshold', 'autocorrelation']]
        ready = np.isfinite(required).all(axis=1)
        eligible = ready & indicators['eruption'].fillna(False)
        direction = np.sign(indicators['return'].fillna(0)) * np.sign(indicators['autocorrelation'].fillna(0))
        entries = direction.where(eligible, 0).fillna(0).astype(int).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        remaining = 0
        for i, entry in enumerate(entries):
            if position:
                remaining -= 1
                if remaining == 0:
                    position = 0
                    continue
            elif entry:
                position = int(entry)
                remaining = params.hold_bars
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame(index=index)
    synthetic = pd.DataFrame({
        'return': [1, -1, -1, 0, -1, 1, 1, 0, 0],
        'volatility': 2.0,
        'threshold': 1.0,
        'autocorrelation': 0.5,
        'eruption': True,
    }, index=index)
    for holding, expected in (
        (1, [0, 1, 0, -1, 0, -1, 0, 1, 0]),
        (2, [0, 1, 1, 0, 0, -1, -1, 0, 0]),
    ):
        result = GeneratedStrategy.generate_signals(data, synthetic, None, VolatilityEchoParams(hold_bars=holding))
        assert result.data['signal'].tolist() == expected
        assert (result.data['size'] > 0).all()
    synthetic['autocorrelation'] = np.nan
    result = GeneratedStrategy.generate_signals(data, synthetic, None, VolatilityEchoParams())
    assert result.data['signal'].eq(0).all()
