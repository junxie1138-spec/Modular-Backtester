from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeParams:
    range_window: int = 12
    overlap_window: int = 6
    atr_window: int = 14
    overlap_cut: float = 0.55
    shock_threshold: float = 1.25
    trail_atr: float = 0.85
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeParams]):
    strategy_id = 'gen_local_1791337885'

    @classmethod
    def params_type(cls) -> type[RangeParams]:
        return RangeParams

    def warmup_bars(self, params: RangeParams) -> int:
        return int(max(params.range_window, params.overlap_window, params.atr_window) + 1)

    def indicators(self, data: pd.DataFrame, params: RangeParams) -> pd.DataFrame:
        if min(params.range_window, params.overlap_window, params.atr_window) < 1:
            raise ValueError('Lookbacks must be positive.')
        if not (0.0 < params.overlap_cut < 1.0):
            raise ValueError('overlap_cut must lie between zero and one.')
        if not (np.isfinite(params.shock_threshold) and params.shock_threshold > 1.0):
            raise ValueError('shock_threshold must exceed one.')
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0.0):
            raise ValueError('trail_atr must be positive.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        high, low, close = data['high'], data['low'], data['close']
        out = pd.DataFrame(index=data.index)
        out['range'] = (high - low).where(high > low)
        out['range_baseline'] = out['range'].shift(1).rolling(params.range_window, min_periods=params.range_window).mean()
        out['range_shock'] = out['range'] / out['range_baseline'].replace(0.0, np.nan)
        out['intersection'] = (np.minimum(high, high.shift(1)) - np.maximum(low, low.shift(1))).clip(lower=0.0)
        out['union'] = (np.maximum(high, high.shift(1)) - np.minimum(low, low.shift(1))).replace(0.0, np.nan)
        out['overlap'] = out['intersection'] / out['union']
        out['regime_overlap'] = out['overlap'].shift(1).rolling(params.overlap_window, min_periods=params.overlap_window).mean()
        out['close_location'] = ((close - low) / out['range']).clip(0.0, 1.0)
        out['midpoint'] = (high + low) / 2.0
        out['midpoint_change'] = out['midpoint'].diff()
        out['previous_low'] = low.shift(1)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = np.isfinite(out[['range_shock', 'regime_overlap', 'close_location', 'midpoint_change', 'previous_low', 'previous_close', 'atr']]).all(axis=1) & (out['atr'] > 0.0)
        migration = (out['regime_overlap'] < params.overlap_cut) & (out['midpoint_change'] > 0.0) & (close > out['previous_close'])
        absorption = (out['regime_overlap'] >= params.overlap_cut) & (low < out['previous_low']) & (close > out['previous_low'])
        out['entry'] = ready & (out['range_shock'] >= params.shock_threshold) & (out['close_location'] >= 0.7) & (migration | absorption)
        out['regime_confidence'] = pd.Series(np.where(migration, (params.overlap_cut - out['regime_overlap']) / params.overlap_cut, (out['regime_overlap'] - params.overlap_cut) / (1.0 - params.overlap_cut)), index=data.index).clip(0.0, 1.0)
        out['shock_confidence'] = ((out['range_shock'] - 1.0) / params.shock_threshold).clip(0.0, 1.0)
        out['strength'] = (out['regime_confidence'] * out['shock_confidence'] * out['close_location']).fillna(0.0).clip(0.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeParams) -> SignalFrame:
        df = pd.DataFrame({'signal': 0, 'size': 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf
        entry_size = 1.0
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        strengths = indicators['strength'].fillna(0.0).to_numpy(dtype=float)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and close > 0.0 and atr > 0.0
            if active:
                held += 1
                if valid:
                    # First update occurs on the session after the entry decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close <= stop or held >= params.hold_bars:
                    active = False
                    continue
            elif valid and entries[i]:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
                entry_size = float(np.clip(0.2 + 0.8 * strengths[i], 0.2, 1.0))
            if active:
                signals[i] = 1
                sizes[i] = entry_size
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100.0, 103.0, 102.0, 102.0, 100.0, 100.0, 101.0, 101.0]}, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, True, False, False, False], 'atr': [0.5, 0.5, 10.0, 0.5, 0.5, 0.5, 0.5, 0.5], 'strength': 0.5}, index=index)
    params = RangeParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 1, 1, 0]
    assert np.allclose(result.loc[result['signal'] == 1, 'size'], 0.6)
    params.hold_bars = 1
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0, 1, 0, 0]


if __name__ == '__main__':
    _self_check()
