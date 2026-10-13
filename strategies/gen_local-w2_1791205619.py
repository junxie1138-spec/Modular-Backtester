from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoveryParams:
    peak_window: int = 20
    atr_window: int = 14
    min_drawdown: float = 0.02
    min_gap: float = 0.001
    breakeven_pct: float = 0.006
    trail_atr: float = 0.8
    initial_stop_atr: float = 1.5

    def __post_init__(self):
        if self.peak_window < 2 or self.atr_window < 2:
            raise ValueError('Lookback windows must be at least two.')
        values = (self.min_drawdown, self.min_gap, self.breakeven_pct,
                  self.trail_atr, self.initial_stop_atr)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Thresholds and stop multipliers must be positive and finite.')


class GeneratedStrategy(BaseStrategy[GapRecoveryParams]):
    strategy_id = 'gen_local-w2_1791205619'

    @classmethod
    def params_type(cls):
        return GapRecoveryParams

    def warmup_bars(self, params: GapRecoveryParams) -> int:
        return max(params.peak_window, params.atr_window) + 2

    def indicators(self, data: pd.DataFrame, params: GapRecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        previous_high = data['high'].shift(1)
        out['previous_close'] = previous_close
        out['previous_high'] = previous_high
        out['peak'] = data['close'].rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out['drawdown'] = data['close'].div(out['peak'].where(out['peak'] > 0)) - 1.0
        out['gap'] = data['open'].div(previous_close.where(previous_close > 0)) - 1.0
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['accepted_gap'] = (
            (out['gap'] >= params.min_gap)
            & (data['open'] > previous_high)
            & (data['close'] >= data['open'])
            & (out['drawdown'].shift(1) <= -params.min_drawdown)
        ).fillna(False)
        out['confirmed'] = (
            out['accepted_gap']
            & out['accepted_gap'].shift(1, fill_value=False)
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapRecoveryParams) -> SignalFrame:
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = peak = 0.0
        held = 0

        for i, (opening, high, low, close) in enumerate(prices):
            if pending:
                pending = False
                # Entry reference is this session's open, observed only now.
                if i == 0 or not np.isfinite(opening) or opening <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    continue
                entry = opening
                peak = opening
                stop = entry - params.initial_stop_atr * atr[i - 1]
                held = 0
                armed = False
                active = True

            if active:
                held += 1
                if not np.isfinite(prices[i]).all() or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                # Test the previously established stop before raising it.
                stopped = opening <= stop or low <= stop
                if not stopped:
                    peak = max(peak, high)
                    if peak >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                        # Same-bar low cannot establish ordering relative to a new stop.
                        stopped = close <= stop
                # All exits are close decisions, executed next session's open.
                if stopped or held >= 2:
                    active = False
                    continue
                signal[i] = 1
            elif confirmed[i] and np.isfinite(prices[i]).all() and np.isfinite(atr[i]) and atr[i] > 0:
                signal[i] = 1
                pending = True

        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    data = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [100.2, 100.2, 101.2, 101.0, 100.2, 100.2],
        'low': [99.8, 99.8, 99.8, 100.0, 99.8, 99.8],
        'close': [100.0, 100.0, 100.8, 100.1, 100.0, 100.0],
        'volume': [1000.0] * 6
    }, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0] * 6,
        'confirmed': [False, True, False, False, False, False]
    }, index=index)
    strategy = GeneratedStrategy()
    params = GapRecoveryParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    # A newly raised stop can exit after the first holding session.
    early = data.copy()
    early.loc[index[2], 'close'] = 100.3
    result = strategy.generate_signals(early, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 0]
    # A surviving trade still exits after its second holding session.
    surviving = data.copy()
    surviving.loc[index[3], ['open', 'high', 'low', 'close']] = [100.9, 101.4, 100.7, 101.2]
    result = strategy.generate_signals(surviving, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    derived = strategy.indicators(data, params)
    assert not derived['confirmed'].any()
    assert strategy.warmup_bars(params) >= max(params.peak_window, params.atr_window) + 2


if __name__ == '__main__':
    _self_check()
