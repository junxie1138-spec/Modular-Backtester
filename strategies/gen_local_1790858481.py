from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_span: int = 12
    slope_span: int = 5
    atr_window: int = 14
    innovation_min: float = 0.015
    position_min: float = 0.15
    position_max: float = 1.5
    stop_atr: float = 1.8
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.trend_span, self.slope_span, self.atr_window) < 2:
            raise ValueError('Indicator periods must be at least two.')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')
        values = [self.innovation_min, self.position_min, self.position_max, self.stop_atr]
        if not np.isfinite(values).all():
            raise ValueError('Thresholds must be finite.')
        if self.innovation_min <= 0 or self.stop_atr <= 0 or not 0 < self.position_min < self.position_max:
            raise ValueError('Invalid innovation, displacement, or stop thresholds.')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790858481'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.trend_span + params.slope_span + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data['close']
        previous = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean().replace(0.0, np.nan)
        center = close.ewm(span=params.trend_span, adjust=False, min_periods=params.trend_span).mean()
        velocity = center.diff()
        slope = velocity.ewm(span=params.slope_span, adjust=False, min_periods=params.slope_span).mean()
        spread = close - center
        return pd.DataFrame({
            'previous_close': previous,
            'true_range': true_range,
            'atr': atr,
            'center': center,
            'velocity': velocity,
            'slope': slope,
            'spread': spread,
            'innovation': slope.diff() / atr,
            'relative_position': spread / atr,
            'spread_change': spread.diff(),
            'price_change': close.diff(),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        setup = (
            (indicators['relative_position'] >= params.position_min)
            & (indicators['relative_position'] <= params.position_max)
            & (indicators['slope'] > 0.0)
            & (indicators['innovation'] >= params.innovation_min)
            & (indicators['spread_change'] < 0.0)
            & (indicators['price_change'] >= 0.0)
            & (indicators['atr'] > 0.0)
        ).fillna(False)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Freeze decision-bar ATR against the next-bar opening entry.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                if (
                    not np.isfinite(stop)
                    or not np.isfinite(closes[i])
                    or closes[i] < stop
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0.0:
                entry_bar = i + 1
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({
        'atr': 2.0,
        'relative_position': [0.5, 0.5] + [0.0] * 8,
        'slope': 0.2,
        'innovation': 0.02,
        'spread_change': -0.1,
        'price_change': 0.1,
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    assert (result['size'] > 0.0).all()
    data.loc[index[2], 'close'] = 95.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    indicators.loc[index[0], 'relative_position'] = 0.0
    assert strategy.generate_signals(data, indicators, ctx, params).data['signal'].eq(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert strategy.generate_signals(data, derived, ctx, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
