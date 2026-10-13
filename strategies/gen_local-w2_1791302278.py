from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 63
    baseline_window: int = 20
    recent_window: int = 5
    atr_window: int = 14
    shift_threshold: float = 1.0
    pressure_threshold: float = 0.15
    min_depth: float = 0.015
    dispersion_floor: float = 0.003
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791302278'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.peak_window + params.baseline_window + params.recent_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if min(p.peak_window, p.baseline_window, p.recent_window, p.atr_window, p.holding_bars) < 1:
            raise ValueError('Windows and holding_bars must be positive')
        if p.baseline_window < 2 or min(p.shift_threshold, p.dispersion_floor, p.breakeven_pct, p.trail_atr) <= 0:
            raise ValueError('Invalid shift, dispersion, breakeven, or trail parameters')
        if not 0 <= p.pressure_threshold < 1 or not 0 <= p.min_depth < 1:
            raise ValueError('Invalid pressure or depth threshold')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['peak'] = close.rolling(p.peak_window, min_periods=p.peak_window).max()
        out['depth'] = 1.0 - close / out['peak']
        out['recent_depth'] = out['depth'].rolling(p.recent_window, min_periods=p.recent_window).mean()
        out['prior_depth'] = out['depth'].shift(p.recent_window).rolling(p.baseline_window, min_periods=p.baseline_window).mean()
        out['prior_dispersion'] = out['depth'].shift(p.recent_window).rolling(p.baseline_window, min_periods=p.baseline_window).std(ddof=0)
        out['depth_shift'] = (out['recent_depth'] - out['prior_depth']) / out['prior_dispersion'].clip(lower=p.dispersion_floor)
        out['bar_range'] = data['high'] - data['low']
        out['close_location'] = ((2.0 * close - data['high'] - data['low']) / out['bar_range'].where(out['bar_range'] > 0)).where(out['bar_range'] != 0, 0.0).clip(-1.0, 1.0)
        out['valid_volume'] = data['volume'].where(data['volume'] >= 0)
        out['signed_volume'] = out['close_location'] * out['valid_volume']
        out['volume_sum'] = out['valid_volume'].rolling(p.recent_window, min_periods=p.recent_window).sum()
        out['signed_volume_sum'] = out['signed_volume'].rolling(p.recent_window, min_periods=p.recent_window).sum()
        out['pressure'] = out['signed_volume_sum'] / out['volume_sum'].where(out['volume_sum'] > 0)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([out['bar_range'], (data['high'] - out['previous_close']).abs(), (data['low'] - out['previous_close']).abs()], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(p.atr_window, min_periods=p.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = out[['depth_shift', 'pressure', 'recent_depth', 'prior_depth', 'atr']].notna().all(axis=1) & out['atr'].gt(0) & close.notna()
        depth_present = out[['recent_depth', 'prior_depth']].max(axis=1).ge(p.min_depth)
        out['long_agreement'] = out['ready'] & depth_present & out['depth_shift'].le(-p.shift_threshold) & out['pressure'].ge(p.pressure_threshold)
        out['short_agreement'] = out['ready'] & depth_present & out['depth_shift'].ge(p.shift_threshold) & out['pressure'].le(-p.pressure_threshold)
        out['long_event'] = out['long_agreement'] & ~out['long_agreement'].shift(1, fill_value=False)
        out['short_event'] = out['short_agreement'] & ~out['short_agreement'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        targets = np.zeros(n, dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        long_event = indicators['long_event'].fillna(False).to_numpy(dtype=bool)
        short_event = indicators['short_event'].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        age = 0
        for i in range(n):
            if side:
                # A preceding close decision becomes a position at this open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                invalid = not (np.isfinite(entry) and entry > 0 and np.isfinite(closes[i]) and closes[i] > 0 and ready[i])
                if invalid:
                    side = 0
                else:
                    # Signed prices make the stop ratchet upward for both sides.
                    signed_close = side * closes[i]
                    signed_entry = side * entry
                    if side * (closes[i] / entry - 1.0) >= params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, signed_entry, signed_close - params.trail_atr * atr[i])
                    # Close-observed exits execute next bar through the shift below.
                    if (armed and signed_close <= stop) or age >= params.holding_bars:
                        side = 0
                targets[i] = side
                continue
            if ready[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = 1 if long_event[i] else (-1 if short_event[i] else 0)
                if side:
                    entry = np.nan
                    stop = -np.inf
                    armed = False
                    age = 0
            targets[i] = side
        df = pd.DataFrame({'signal': targets, 'size': np.ones(n, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': [100.0, 103.0, 100.0, 100.0, 100.0, 97.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'atr': 1.0, 'ready': True, 'long_event': [True, False, False, False, False, False, False, False, False], 'short_event': [False, False, False, False, True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
    assert result['size'].gt(0).all()
    timed = strategy.generate_signals(data, derived, None, Params(holding_bars=2, breakeven_pct=0.5)).data
    assert timed['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
    prefix = strategy.generate_signals(data.iloc[:6], derived.iloc[:6], None, Params()).data
    assert prefix.equals(result.iloc[:6])


if __name__ == '__main__':
    _self_check()
