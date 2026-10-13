from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_bars: int = 5
    smooth_bars: int = 20
    correlation_bars: int = 63
    percentile_bars: int = 126
    entry_quantile: float = 0.60


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = 'gen_local-w2_1791330038'

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    @staticmethod
    def warmup_bars(params: AccelerationParams) -> int:
        return int(3 * params.roc_bars + params.smooth_bars + params.correlation_bars + params.percentile_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        if min(params.roc_bars, params.smooth_bars, params.percentile_bars) < 1 or params.correlation_bars < 2:
            raise ValueError('Lookbacks must be positive; correlation_bars must be at least two.')
        if not 0.0 < params.entry_quantile < 1.0:
            raise ValueError('entry_quantile must lie strictly between zero and one.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['roc'] = close.div(close.shift(params.roc_bars)).sub(1.0)
        out['acceleration'] = out['roc'].diff(params.roc_bars)
        out['smoothed_acceleration'] = out['acceleration'].rolling(params.smooth_bars, min_periods=params.smooth_bars).mean()
        out['lagged_acceleration'] = out['smoothed_acceleration'].shift(params.roc_bars)
        out['autocorrelation'] = out['smoothed_acceleration'].rolling(params.correlation_bars, min_periods=params.correlation_bars).corr(out['lagged_acceleration']).clip(-1.0, 1.0)
        out['forecast'] = out['smoothed_acceleration'] * out['autocorrelation']
        out['historical_forecast'] = out['forecast'].shift(1)
        out['threshold'] = out['historical_forecast'].rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.entry_quantile)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        valid = indicators[['forecast', 'threshold']].notna().all(axis=1)
        entry_condition = valid & (indicators['forecast'] > indicators['threshold']) & (indicators['forecast'] > 0.0)
        df = pd.DataFrame(index=data.index)
        # Reversal exit: exposure becomes zero whenever the entry predicate flips false.
        df['signal'] = entry_condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = AccelerationParams()
    rng = np.random.default_rng(19)
    index = pd.date_range('2020-01-01', periods=500, freq='B')
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(index))))
    data = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close, 'volume': 1000000.0}, index=index)
    indicators = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    condition = indicators[['forecast', 'threshold']].notna().all(axis=1) & (indicators['forecast'] > indicators['threshold']) & (indicators['forecast'] > 0.0)
    expected = condition.astype(int).shift(1).fillna(0).astype(int)
    assert signals['signal'].equals(expected)
    assert set(signals['signal'].unique()) <= {0, 1}
    assert (signals['size'] > 0.0).all()
    assert signals.index.equals(data.index)
    first_valid = indicators['threshold'].first_valid_index()
    assert first_valid is not None
    assert data.index.get_loc(first_valid) < GeneratedStrategy.warmup_bars(params)
    prefix = GeneratedStrategy.indicators(data.iloc[:350], params)
    pd.testing.assert_frame_equal(indicators.iloc[:350], prefix)
    flat = data.copy()
    flat[['open', 'high', 'low', 'close']] = 100.0
    flat_indicators = GeneratedStrategy.indicators(flat, params)
    flat_signals = GeneratedStrategy.generate_signals(flat, flat_indicators, None, params).data
    assert flat_signals['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
