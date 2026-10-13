from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EscapeParams:
    breakout_window: int = 20
    rank_window: int = 60
    rejection_window: int = 5
    rejection_min: float = 0.25
    volume_rank_max: float = 0.40
    atr_window: int = 14
    trail_atr: float = 1.5
    hold_bars: int = 2

    def __post_init__(self):
        windows = (self.breakout_window, self.rank_window, self.rejection_window, self.atr_window)
        if any(not isinstance(w, int) or isinstance(w, bool) or w < 2 for w in windows):
            raise ValueError('Lookbacks must be integers of at least two.')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        if not 0.0 <= self.rejection_min <= 1.0:
            raise ValueError('rejection_min must be between zero and one.')
        if not 0.0 < self.volume_rank_max <= 1.0:
            raise ValueError('volume_rank_max must be between zero and one.')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0.0:
            raise ValueError('trail_atr must be finite and positive.')


class GeneratedStrategy(BaseStrategy[EscapeParams]):
    strategy_id = 'gen_local_1790911280'

    @classmethod
    def params_type(cls):
        return EscapeParams

    @staticmethod
    def warmup_bars(params):
        return int(max(200, params.breakout_window + 1, params.rank_window,
                       params.rejection_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['bar_range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['upper_wick'] = ((data['high'] - data[['open', 'close']].max(axis=1))
                             / out['bar_range']).clip(0.0, 1.0)
        out['close_location'] = ((data['close'] - data['low'])
                                 / out['bar_range']).clip(0.0, 1.0)
        out['rejection'] = out['upper_wick'].shift(1).rolling(
            params.rejection_window, min_periods=params.rejection_window).mean()
        out['resistance'] = data['high'].shift(1).rolling(
            params.breakout_window, min_periods=params.breakout_window).max()
        out['volume_rank'] = data['volume'].where(data['volume'] > 0.0).rolling(
            params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        required = ['rejection', 'resistance', 'volume_rank', 'ma200', 'close_location', 'atr']
        valid = pd.Series(np.isfinite(indicators[required].to_numpy()).all(axis=1),
                          index=data.index)
        valid &= np.isfinite(data['close']) & (indicators['atr'] > 0.0)
        entries = (valid
                   & (data['close'] > indicators['ma200'])
                   & (data['close'] > indicators['resistance'])
                   & (indicators['rejection'] >= params.rejection_min)
                   & (indicators['volume_rank'] <= params.volume_rank_max)
                   & (indicators['close_location'] >= 0.80)).fillna(False)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        eligible = entries.to_numpy(dtype=bool)
        usable = valid.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        highest_close = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if active:
                age += 1
                if not usable[i] or not np.isfinite(opens[i]):
                    active = False
                    continue
                # The entry fills on this bar; exclude the preceding decision close.
                if age == 1:
                    stop = opens[i] - params.trail_atr * atrs[i - 1]
                highest_close = max(highest_close, closes[i])
                stop = max(stop, highest_close - params.trail_atr * atrs[i])
                if closes[i] <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif eligible[i]:
                active = True
                age = 0
                highest_close = -np.inf
                stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': [100.0, 100.0, 100.0, 100.0, 101.0, 102.0],
        'close': [100.0, 98.0, 100.0, 101.0, 102.0, 103.0]
    }, index=index)
    indicators = pd.DataFrame({
        'rejection': 0.30, 'resistance': 99.0, 'volume_rank': 0.20,
        'ma200': 90.0, 'close_location': 0.90, 'atr': 1.0
    }, index=index)
    params = EscapeParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 1, 1, 0]
    assert (result['size'] > 0.0).all()
    for n in range(1, len(data) + 1):
        prefix = GeneratedStrategy.generate_signals(
            data.iloc[:n], indicators.iloc[:n], None, params).data
        assert prefix['signal'].equals(result['signal'].iloc[:n])
