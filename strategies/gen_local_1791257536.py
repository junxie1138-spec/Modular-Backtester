from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuietStreakParams:
    streak_length: int = 3
    pressure_window: int = 10
    pressure_threshold: float = 0.15
    vol_window: int = 20
    atr_window: int = 14
    target_vol: float = 0.12
    trail_atr: float = 3.0
    hold_bars: int = 18
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[QuietStreakParams]):
    strategy_id = 'gen_local_1791257536'

    @classmethod
    def params_type(cls) -> type[QuietStreakParams]:
        return QuietStreakParams

    def warmup_bars(self, params: QuietStreakParams) -> int:
        return int(max(params.vol_window + params.streak_length + 1,
                       params.atr_window + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: QuietStreakParams) -> pd.DataFrame:
        p = params
        if min(p.vol_window, p.atr_window, p.pressure_window) < 2:
            raise ValueError('Rolling windows must be at least two.')
        if p.streak_length < 1 or p.hold_bars < 1:
            raise ValueError('Streak and holding lengths must be positive.')
        if not (0 < p.pressure_threshold < 1 and 0 < p.max_size <= 1):
            raise ValueError('Invalid pressure threshold or maximum size.')
        if not (np.isfinite(p.target_vol) and p.target_vol > 0
                and np.isfinite(p.trail_atr) and p.trail_atr > 0):
            raise ValueError('Volatility target and ATR multiplier must be positive and finite.')
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['return'] = data['close'].pct_change(fill_method=None)
        out['daily_vol'] = out['return'].rolling(p.vol_window).std(ddof=0)
        out['quiet_limit'] = out['daily_vol'].shift(1)
        out['annual_vol'] = out['daily_vol'] * np.sqrt(252.0)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_window).mean()
        out['quiet_up'] = (out['return'].gt(0)
                           & out['return'].le(out['quiet_limit'])).fillna(False)
        out['quiet_down'] = (out['return'].lt(0)
                             & out['return'].ge(-out['quiet_limit'])).fillna(False)
        for direction in ('up', 'down'):
            mask = out['quiet_' + direction]
            out[direction + '_streak'] = mask.astype(int).groupby((~mask).cumsum()).cumsum()
        out['bar_range'] = (data['high'] - data['low']).replace(0, np.nan)
        out['auction_pressure'] = ((data['close'] - data['open'])
                                   / out['bar_range']).clip(-1, 1).fillna(0.0)
        out['usable_volume'] = data['volume'].clip(lower=0).fillna(0.0)
        out['weighted_pressure'] = out['auction_pressure'] * out['usable_volume']
        out['pressure_sum'] = out['weighted_pressure'].rolling(p.pressure_window).sum()
        out['volume_sum'] = out['usable_volume'].rolling(p.pressure_window).sum()
        out['pressure'] = out['pressure_sum'] / out['volume_sum'].replace(0, np.nan)
        ready = (np.isfinite(out['atr']) & out['atr'].gt(0)
                 & np.isfinite(out['annual_vol']) & out['annual_vol'].gt(0)
                 & np.isfinite(out['pressure']) & np.isfinite(data['close'])
                 & data['close'].gt(0))
        long_entry = (ready & out['up_streak'].ge(p.streak_length)
                      & out['pressure'].ge(p.pressure_threshold))
        short_entry = (ready & out['down_streak'].ge(p.streak_length)
                       & out['pressure'].le(-p.pressure_threshold))
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out['risk_size'] = (p.target_vol / out['annual_vol'].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).clip(lower=min(0.01, p.max_size), upper=p.max_size).fillna(
            min(0.01, p.max_size))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: QuietStreakParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        position = 0
        entry_bar = 0
        high_water = low_water = stop = 0.0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not (np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0):
                position = 0
                continue
            if position:
                if position == 1:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = close < stop
                else:
                    # Mirrored short exit: the lowest close anchors a downward-ratcheting stop.
                    low_water = min(low_water, close)
                    stop = min(stop, low_water + params.trail_atr * atr)
                    stopped = close > stop
                if stopped or i - entry_bar >= params.hold_bars:
                    position = 0
                    continue
            elif entries[i]:
                position = int(entries[i])
                entry_bar = i
                high_water = low_water = close
                stop = close - position * params.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = indicators['risk_size'].shift(1).fillna(min(0.01, params.max_size)).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    path = np.array([100.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0])
    for side in (1, -1):
        close = path if side == 1 else 200.0 - path
        data = pd.DataFrame({'open': close, 'high': close + 0.5,
                             'low': close - 0.5, 'close': close, 'volume': 1.0}, index=index)
        derived = pd.DataFrame({'atr': [1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                                'entry': [side, 0, 0, 0, 0, 0, 0],
                                'risk_size': 0.4}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, derived, None, QuietStreakParams()).data
        assert result['signal'].tolist() == [0, side, side, side, side, side, 0]
        assert result['size'].gt(0).all()
        result = GeneratedStrategy.generate_signals(None, data, derived, None,
                                                    QuietStreakParams(hold_bars=3)).data
        assert result['signal'].tolist() == [0, side, side, side, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
