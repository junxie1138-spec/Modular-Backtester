from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    streak_bars: int = 3
    excursion_window: int = 10
    excursion_edge: float = 0.20
    profit_target: float = 0.045
    max_hold: int = 18
    size: float = 1.0

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in
               (self.streak_bars, self.excursion_window, self.max_hold)):
            raise ValueError('Lookbacks and max_hold must be positive integers')
        if not 0.0 < self.excursion_edge < 1.0:
            raise ValueError('excursion_edge must lie strictly between zero and one')
        if not all(np.isfinite(x) and x > 0.0 for x in
                   (self.profit_target, self.size)):
            raise ValueError('profit_target and size must be finite and positive')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1790896642'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.excursion_window, params.streak_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        high = data['high'].replace([np.inf, -np.inf], np.nan)
        low = data['low'].replace([np.inf, -np.inf], np.nan)
        out['previous_close'] = close.shift(1)
        out['close_change'] = close.diff()
        rising = out['close_change'].gt(0)
        falling = out['close_change'].lt(0)
        out['up_streak'] = rising.astype(int).groupby((~rising).cumsum()).cumsum().clip(upper=params.streak_bars)
        out['down_streak'] = falling.astype(int).groupby((~falling).cumsum()).cumsum().clip(upper=params.streak_bars)
        out['up_excursion'] = (high - out['previous_close']).clip(lower=0)
        out['down_excursion'] = (out['previous_close'] - low).clip(lower=0)
        out['up_total'] = out['up_excursion'].rolling(params.excursion_window).sum()
        out['down_total'] = out['down_excursion'].rolling(params.excursion_window).sum()
        out['excursion_total'] = out['up_total'] + out['down_total']
        out['excursion_balance'] = ((out['up_total'] - out['down_total']) /
                                    out['excursion_total'].where(out['excursion_total'] > 0))
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        balance = indicators['excursion_balance']
        long_entry = (indicators['up_streak'].ge(params.streak_bars) &
                      balance.ge(params.excursion_edge))
        short_entry = (indicators['down_streak'].ge(params.streak_bars) &
                       balance.le(-params.excursion_edge))
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)],
                            [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        direction = 0
        decision_bar = -1
        entry_price = np.nan
        ready_bar = self.warmup_bars(params) - 1

        for i in range(len(data)):
            if direction:
                # Read the entry open only when that bar has actually arrived.
                if i == decision_bar + 1:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    direction = 0
                else:
                    gain = direction * (closes[i] / entry_price - 1.0)
                    held_bars = i - decision_bar
                    if (np.isfinite(gain) and gain >= params.profit_target) or held_bars >= params.max_hold:
                        direction = 0
                # An exit stays flat for this decision; no same-bar re-entry.
            elif i >= ready_bar and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(entries[i])
                if direction:
                    decision_bar = i
                    entry_price = np.nan
            raw[i] = direction

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=16, freq='D')
    for direction in (1, -1):
        close = 100.0 + direction * np.arange(len(index), dtype=float)
        data = pd.DataFrame({'open': close, 'high': close + 0.1,
                             'low': close - 0.1, 'close': close,
                             'volume': 1000.0}, index=index)
        for target, exit_bar in ((1.0, 8), (0.005, 6)):
            params = GeneratedParams(streak_bars=2, excursion_window=3,
                                     profit_target=target, max_hold=4)
            ctx = StrategyContext(symbol='SPY', timeframe='1d',
                                  warmup_bars=strategy.warmup_bars(params))
            features = strategy.indicators(data, params)
            signals = strategy.generate_signals(data, features, ctx, params).data
            assert signals['signal'].iloc[:4].eq(0).all()
            assert signals['signal'].iloc[4:exit_bar].eq(direction).all()
            assert signals['signal'].iloc[exit_bar] == 0
            assert signals['size'].gt(0).all()
            assert set(signals['signal'].unique()) <= {-1, 0, 1}
            blocked = features.copy()
            blocked['excursion_balance'] = -direction
            assert strategy.generate_signals(data, blocked, ctx, params).data['signal'].eq(0).all()
            prefix = data.iloc[:9]
            prefix_signals = strategy.generate_signals(
                prefix, strategy.indicators(prefix, params), ctx, params).data
            pd.testing.assert_frame_equal(signals.iloc[:9], prefix_signals)
    empty = data.iloc[:0]
    assert strategy.generate_signals(
        empty, strategy.indicators(empty, params), ctx, params).data.empty


if __name__ == '__main__':
    _self_check()
