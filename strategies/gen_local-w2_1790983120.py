from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    peak_window: int = 63
    baseline_window: int = 20
    reaction_window: int = 3
    min_depth: float = 0.02
    depth_shift: float = 0.0015
    pressure_shift: float = 0.20
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.peak_window, self.baseline_window, self.reaction_window, self.hold_bars) < 1:
            raise ValueError('Windows and holding period must be positive.')
        if not (0.0 < self.min_depth < 1.0):
            raise ValueError('min_depth must lie between zero and one.')
        if not np.isfinite(self.depth_shift) or self.depth_shift <= 0.0:
            raise ValueError('depth_shift must be finite and positive.')
        if not np.isfinite(self.pressure_shift) or not (0.0 < self.pressure_shift <= 2.0):
            raise ValueError('pressure_shift must lie between zero and two.')


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = 'gen_local-w2_1790983120'

    @classmethod
    def params_type(cls):
        return ShockwaveParams

    def warmup_bars(self, params: ShockwaveParams) -> int:
        return int(params.peak_window + params.baseline_window + params.reaction_window + 1)

    def indicators(self, data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        out['peak'] = data['high'].rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = 1.0 - close / out['peak'].where(out['peak'] > 0.0)
        # Ignore peak changes so an expiring historical high cannot mimic repair.
        out['depth_velocity'] = out['drawdown'].diff().where(out['peak'].eq(out['peak'].shift(1)))
        out['recent_velocity'] = out['depth_velocity'].rolling(params.reaction_window, min_periods=params.reaction_window).mean()
        out['prior_velocity'] = out['depth_velocity'].rolling(params.baseline_window, min_periods=params.baseline_window).mean().shift(params.reaction_window)
        out['depth_break'] = out['recent_velocity'] - out['prior_velocity']
        out['bar_range'] = data['high'] - data['low']
        out['close_location'] = ((2.0 * close - data['high'] - data['low']) / out['bar_range'].where(out['bar_range'] > 0.0)).clip(-1.0, 1.0)
        out.loc[out['bar_range'].eq(0.0) & close.notna(), 'close_location'] = 0.0
        out['valid_volume'] = data['volume'].where(data['volume'] >= 0.0)
        out['signed_volume'] = out['close_location'] * out['valid_volume']
        out['recent_volume'] = out['valid_volume'].rolling(params.reaction_window, min_periods=params.reaction_window).sum()
        out['prior_volume'] = out['valid_volume'].rolling(params.baseline_window, min_periods=params.baseline_window).sum().shift(params.reaction_window)
        out['recent_signed_volume'] = out['signed_volume'].rolling(params.reaction_window, min_periods=params.reaction_window).sum()
        out['prior_signed_volume'] = out['signed_volume'].rolling(params.baseline_window, min_periods=params.baseline_window).sum().shift(params.reaction_window)
        out['recent_pressure'] = out['recent_signed_volume'] / out['recent_volume'].where(out['recent_volume'] > 0.0)
        out['prior_pressure'] = out['prior_signed_volume'] / out['prior_volume'].where(out['prior_volume'] > 0.0)
        out['pressure_break'] = out['recent_pressure'] - out['prior_pressure']
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockwaveParams) -> SignalFrame:
        x = indicators
        required = ['drawdown', 'recent_velocity', 'prior_velocity', 'depth_break', 'recent_pressure', 'prior_pressure', 'pressure_break']
        ready = np.isfinite(x[required]).all(axis=1) & x['drawdown'].ge(params.min_depth)
        ready &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        depth_long = x['prior_velocity'].gt(0.0) & x['recent_velocity'].lt(0.0) & x['depth_break'].le(-params.depth_shift)
        depth_short = x['prior_velocity'].lt(0.0) & x['recent_velocity'].gt(0.0) & x['depth_break'].ge(params.depth_shift)
        pressure_long = x['prior_pressure'].lt(0.0) & x['recent_pressure'].gt(0.0) & x['pressure_break'].ge(params.pressure_shift)
        pressure_short = x['prior_pressure'].gt(0.0) & x['recent_pressure'].lt(0.0) & x['pressure_break'].le(-params.pressure_shift)
        entries = np.select([ready & depth_long & pressure_long, ready & depth_short & pressure_short], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(entries):
            if position:
                if i == exit_bar:
                    position = 0
            elif candidate:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = ShockwaveParams()
    strategy = GeneratedStrategy()
    n = strategy.warmup_bars(params) + 25
    index = pd.date_range('2020-01-01', periods=n, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    x = pd.DataFrame(np.nan, index=index, columns=['drawdown', 'recent_velocity', 'prior_velocity', 'depth_break', 'recent_pressure', 'prior_pressure', 'pressure_break'])
    entry = strategy.warmup_bars(params)
    x.iloc[entry] = [0.05, -0.002, 0.002, -0.004, 0.4, -0.4, 0.8]
    x.iloc[entry + 1] = [0.05, 0.002, -0.002, 0.004, -0.4, 0.4, -0.8]
    x.iloc[entry + 11] = x.iloc[entry + 1]
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    actual = strategy.generate_signals(data, x, ctx, params).data
    expected = np.zeros(n, dtype=int)
    expected[entry + 1:entry + 1 + params.hold_bars] = 1
    expected[entry + 12:entry + 12 + params.hold_bars] = -1
    assert np.array_equal(actual['signal'].to_numpy(), expected)
    assert actual['size'].gt(0.0).all()
    x.iloc[entry, x.columns.get_loc('recent_pressure')] = -0.4
    assert strategy.generate_signals(data, x, ctx, params).data['signal'].iloc[entry + 1] == 0
    derived = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-5], params)
    pd.testing.assert_frame_equal(derived.iloc[:-5], prefix)


if __name__ == '__main__':
    _self_check()
