from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    fractional_order: float = 0.45
    fractional_terms: int = 64
    center_window: int = 20
    correlation_window: int = 60
    cycle_lag: int = 18
    percentile_window: int = 126
    entry_percentile: float = 0.80
    atr_window: int = 14
    trailing_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = 'gen_local_1790962214'

    @classmethod
    def params_type(cls):
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(
            params.fractional_terms + params.center_window + params.cycle_lag
            + params.correlation_window + params.percentile_window,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if min(params.fractional_terms, params.center_window,
               params.correlation_window, params.percentile_window,
               params.atr_window) < 2 or params.cycle_lag < 2:
            raise ValueError('Indicator windows and cycle_lag must be at least two')
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between zero and one')
        if not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0.0 or params.holding_bars < 1:
            raise ValueError('Trailing multiplier and holding period must be positive')
        close = data['close'].astype(float)
        log_close = np.log(close.where(close > 0.0))
        weights = np.empty(params.fractional_terms, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_terms):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional = pd.Series(0.0, index=data.index)
        for lag, weight in enumerate(weights):
            fractional = fractional + weight * log_close.shift(lag)
        center = fractional.rolling(params.center_window).mean()
        oscillator = fractional - center
        full = oscillator.rolling(params.correlation_window).corr(oscillator.shift(params.cycle_lag))
        half = oscillator.rolling(params.correlation_window).corr(oscillator.shift(params.cycle_lag // 2))
        resonance = (full - half).replace([np.inf, -np.inf], np.nan)
        threshold = resonance.shift(1).rolling(params.percentile_window).quantile(params.entry_percentile)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame({
            'fractional_close': fractional,
            'fractional_center': center,
            'oscillator': oscillator,
            'full_cycle_correlation': full,
            'half_cycle_correlation': half,
            'resonance': resonance,
            'resonance_threshold': threshold,
            'true_range': true_range,
            'atr': atr,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        oscillator = indicators['oscillator']
        entries = (
            (oscillator > 0.0)
            & (oscillator.shift(1) <= 0.0)
            & (indicators['resonance'] > indicators['resonance_threshold'])
        ).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        peak = np.nan
        stop = -np.inf
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if not valid:
                active = False
                age = 0
                peak = np.nan
                stop = -np.inf
                continue
            if active:
                # Entry fills on this bar after the final one-bar signal shift.
                peak = close if np.isnan(peak) else max(peak, close)
                stop = max(stop, peak - params.trailing_atr * atr)
                age += 1
                if close <= stop or age >= params.holding_bars:
                    active = False
                    age = 0
                    peak = np.nan
                    stop = -np.inf
            elif entries[i]:
                active = True
                age = 0
                peak = np.nan
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveParams()
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    close = np.array([100.0, 100.0, 100.0, 110.0, 105.0, 104.0])
    data = pd.DataFrame({'open': close, 'high': close + 1.0,
                         'low': close - 1.0, 'close': close,
                         'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({
        'oscillator': [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'resonance': 2.0, 'resonance_threshold': 1.0,
        'atr': [1.0, 1.0, 1.0, 1.0, 3.0, 3.0],
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0]
    assert (result['size'] > 0.0).all()
    params.holding_bars = 2
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    params.holding_bars = 18
    t = np.arange(400, dtype=float)
    prices = 100.0 * np.exp(0.0003 * t + 0.02 * np.sin(2.0 * np.pi * t / 18.0))
    history = pd.DataFrame({'open': prices, 'high': prices + 1.0,
                            'low': prices - 1.0, 'close': prices,
                            'volume': 1000.0},
                           index=pd.date_range('2020-01-01', periods=400, freq='B'))
    complete = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:320], params)
    pd.testing.assert_frame_equal(complete.iloc[:320], prefix)
    assert complete['resonance_threshold'].iloc[strategy.warmup_bars(params) - 1:].notna().all()
    full_signals = strategy.generate_signals(history, complete, ctx, params).data
    prefix_signals = strategy.generate_signals(history.iloc[:320], prefix, ctx, params).data
    pd.testing.assert_frame_equal(full_signals.iloc[:320], prefix_signals)


if __name__ == '__main__':
    _self_check()
