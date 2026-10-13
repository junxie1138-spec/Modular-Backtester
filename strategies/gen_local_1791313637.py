from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResilienceParams:
    peak_window: int = 63
    envelope_window: int = 21
    min_depth: float = 0.012
    contraction_ratio: float = 0.75
    profit_target: float = 0.045
    hold_bars: int = 18

    def __post_init__(self):
        if self.peak_window < 2 or self.envelope_window < 2 or self.hold_bars < 1:
            raise ValueError('Lookbacks must be at least two and hold_bars positive.')
        if not 0.0 < self.min_depth < 1.0:
            raise ValueError('min_depth must be between zero and one.')
        if not 0.0 < self.contraction_ratio < 1.0:
            raise ValueError('contraction_ratio must be between zero and one.')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0.0:
            raise ValueError('profit_target must be finite and positive.')


class GeneratedStrategy(BaseStrategy[ResilienceParams]):
    strategy_id = 'gen_local_1791313637'

    @classmethod
    def params_type(cls):
        return ResilienceParams

    @staticmethod
    def warmup_bars(params):
        return int(params.peak_window + 2 * params.envelope_window + 1)

    @staticmethod
    def indicators(data, params):
        close = data['close'].astype(float).where(lambda x: np.isfinite(x) & (x > 0.0))
        out = pd.DataFrame(index=data.index)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['depth'] = (1.0 - close / out['peak']).clip(lower=0.0)
        out['depth_lag'] = out['depth'].shift(1)
        out['peak_lag'] = out['peak'].shift(params.envelope_window)
        out['recent_worst'] = out['depth'].rolling(params.envelope_window, min_periods=params.envelope_window).max()
        out['prior_worst'] = out['recent_worst'].shift(params.envelope_window)
        out['retention'] = out['recent_worst'] / out['prior_worst'].replace(0.0, np.nan)
        out['entry'] = (
            (out['peak'] > out['peak_lag'])
            & (out['depth'] >= params.min_depth)
            & (out['depth_lag'] < params.min_depth)
            & (out['retention'] <= params.contraction_ratio)
        ).fillna(False)
        out['strength'] = (
            (out['depth'] / params.min_depth)
            * (1.0 - out['retention'])
            / (1.0 - params.contraction_ratio)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.25, dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strength = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.25
        for i in range(n):
            if active:
                if i == entry_bar:
                    # The entry was requested yesterday; today's open is now observable.
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                        continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_hit = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_hit:
                    # A close-based exit decision becomes a flat target next bar.
                    active = False
                    continue
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                held_size = float(0.25 + 0.70 * strength[i])
            if active:
                signal[i] = 1
                size[i] = held_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': [100.0, 110.0, 111.0, 112.0, 113.0, 114.0], 'close': [100.0, 110.0, 111.0, 116.0, 113.0, 114.0]}, index=index)
    ind = pd.DataFrame({'entry': [True, False, False, False, False, False], 'strength': [0.5] * 6}, index=index)
    strategy = GeneratedStrategy
    result = strategy.generate_signals(data, ind, None, ResilienceParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    assert np.allclose(result.loc[result['signal'] == 1, 'size'], 0.60)
    timed = strategy.generate_signals(data, ind, None, ResilienceParams(profit_target=1.0, hold_bars=2)).data
    assert timed['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    quiet = strategy.indicators(data, ResilienceParams())
    assert not quiet['entry'].any()
    assert strategy.generate_signals(data, quiet, None, ResilienceParams()).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
