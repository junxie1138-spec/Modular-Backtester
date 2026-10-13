from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakSeasonalityParams:
    atr_period: int = 14
    min_streak: int = 3
    stop_atr: float = 2.0
    hold_bars: int = 18

    def __post_init__(self):
        if self.atr_period < 1 or self.min_streak < 1 or self.hold_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')


class GeneratedStrategy(BaseStrategy[StreakSeasonalityParams]):
    strategy_id = 'gen_local_1791294886'

    @classmethod
    def params_type(cls):
        return StreakSeasonalityParams

    def warmup_bars(self, params):
        return int(max(params.atr_period + 1, params.min_streak + 3))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out['change'] = data['close'].diff()
        out['down'] = out['change'].lt(0) & out['change'].notna()
        out['up'] = out['change'].gt(0) & out['change'].notna()
        # Count consecutive declines, capped at the entry threshold.
        consecutive = pd.Series(True, index=data.index)
        streak = pd.Series(0, index=data.index, dtype=int)
        for lag in range(params.min_streak):
            consecutive = consecutive & out['down'].shift(lag, fill_value=False)
            streak = streak + consecutive.astype(int)
        out['down_streak'] = streak
        out['two_bar_confirmation'] = (
            out['up'] & out['up'].shift(1, fill_value=False)
        )
        out['seasonal_window'] = (data.index.day >= 10) & (data.index.day <= 20)
        out['entry_ready'] = (
            out['down_streak'].shift(2).ge(params.min_streak)
            & out['two_bar_confirmation']
            & out['seasonal_window']
            & out['atr'].gt(0)
            & np.isfinite(out['atr'])
            & data['close'].gt(0)
            & np.isfinite(data['close'])
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry_ready'].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        first_ready = max(params.atr_period + 1, params.min_streak + 3) - 1

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Observe the fill bar's open only when that bar is reached.
                    if np.isfinite(opens[i]) and opens[i] > 0:
                        fixed_stop = opens[i] - params.stop_atr * entry_atr
                    else:
                        active = False
                stop_hit = np.isfinite(closes[i]) and closes[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.hold_bars
                if stop_hit or expired:
                    active = False
            elif i >= first_ready and entries[i]:
                active = True
                entry_bar = i + 1
                entry_atr = atr[i]
                fixed_stop = np.nan
            decisions[i] = int(active)

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = StreakSeasonalityParams(atr_period=3)
    index = pd.bdate_range('2020-01-06', periods=35)
    close = np.array([100, 99, 98, 97, 98, 99] + [99] * 29, dtype=float)
    data = pd.DataFrame({
        'open': close,
        'high': close + 1,
        'low': close - 1,
        'close': close,
        'volume': 1000000.0,
    }, index=index)
    derived = GeneratedStrategy.indicators(None, data, params)
    assert derived['entry_ready'].iloc[5]
    assert not derived['entry_ready'].iloc[:5].any()
    result = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    assert result['signal'].iloc[:6].eq(0).all()
    assert result['signal'].iloc[6:24].eq(1).all()
    assert result['signal'].iloc[24:].eq(0).all()
    data.loc[index[8], ['open', 'high', 'low', 'close']] = [94, 95, 93, 94]
    derived = GeneratedStrategy.indicators(None, data, params)
    stopped = GeneratedStrategy.generate_signals(None, data, derived, None, params).data
    assert stopped['signal'].iloc[8] == 1
    assert stopped['signal'].iloc[9] == 0
    assert stopped['signal'].isin([0, 1]).all()
    assert stopped['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
