from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    path_window: int = 8
    breakout_window: int = 20
    volume_window: int = 20
    area_threshold: float = 0.06
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1791336524'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.volume_window + params.path_window,
                       params.path_window + 1,
                       params.breakout_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.path_window, params.breakout_window, params.volume_window) < 2:
            raise ValueError('Lookback windows must be at least two bars')
        if params.hold_bars < 1:
            raise ValueError('hold_bars must be positive')
        if not 0.0 < params.area_threshold < 0.5:
            raise ValueError('area_threshold must lie between zero and 0.5')
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        volume = data['volume'].where(data['volume'] >= 0)
        out['volume_baseline'] = volume.rolling(params.volume_window).mean().shift(1)
        out['relative_volume'] = volume / out['volume_baseline'].where(out['volume_baseline'] > 0)
        r = out['return']
        v = out['relative_volume']
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature: half the sum over i < j.
        for lag in range(1, params.path_window):
            area += (v.shift(lag) * r - r.shift(lag) * v).rolling(params.path_window - lag).sum()
        out['raw_area'] = 0.5 * area
        out['return_variation'] = r.abs().rolling(params.path_window).sum()
        out['participation'] = v.rolling(params.path_window).sum()
        out['area_scale'] = out['return_variation'] * out['participation']
        out['area'] = (out['raw_area'] / out['area_scale'].where(out['area_scale'] > 0)).clip(-0.5, 0.5)
        out['upper'] = data['high'].rolling(params.breakout_window).max().shift(1)
        out['lower'] = data['low'].rolling(params.breakout_window).min().shift(1)
        out['conviction'] = (out['area'].abs() / params.area_threshold - 1.0).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        area = indicators['area']
        valid = np.isfinite(indicators[['area', 'upper', 'lower', 'conviction']]).all(axis=1)
        valid &= np.isfinite(data['close'])
        long_entry = valid & (data['close'] > indicators['upper']) & (area > params.area_threshold)
        short_entry = valid & (data['close'] < indicators['lower']) & (area < -params.area_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        conviction = indicators['conviction'].fillna(0.0).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        side = 0
        expiry = -1
        entry_size = params.min_size
        for i in range(len(data)):
            if side:
                if i >= expiry:
                    side = 0
                else:
                    signals[i] = side
                    sizes[i] = entry_size
                # Leave an explicit flat bar on expiry; ignore all entry signals.
                continue
            if entries[i]:
                side = int(entries[i])
                expiry = i + params.hold_bars
                entry_size = float(params.min_size + (params.max_size - params.min_size) * conviction[i])
                signals[i] = side
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GeneratedParams()
    n = 2 * params.hold_bars + 4
    index = pd.date_range('2020-01-01', periods=n)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'area': 0.2, 'upper': 90.0, 'lower': 110.0,
                               'conviction': 0.5}, index=index)
    indicators.iloc[params.hold_bars + 1:, indicators.columns.get_loc('area')] = -0.2
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = GeneratedStrategy().generate_signals(data, indicators, ctx, params).data
    h = params.hold_bars
    assert result['signal'].iloc[0] == 0
    assert (result['signal'].iloc[1:h + 1] == 1).all()
    assert result['signal'].iloc[h + 1] == 0
    assert (result['signal'].iloc[h + 2:2 * h + 2] == -1).all()
    assert result['signal'].iloc[2 * h + 2] == 0
    expected_size = params.min_size + 0.5 * (params.max_size - params.min_size)
    assert np.allclose(result['size'].iloc[1:h + 1], expected_size)
    indicators['conviction'] = 1.0
    larger = GeneratedStrategy().generate_signals(data, indicators, ctx, params).data
    assert larger['size'].iloc[1] > result['size'].iloc[1]
    actual = GeneratedStrategy().indicators(data, params)
    assert actual['area'].isna().all()
    assert actual.index.equals(data.index)


if __name__ == '__main__':
    _self_check()
