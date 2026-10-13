from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RevivalParams:
    memory_window: int = 15
    revival_window: int = 4
    min_revivals: int = 1
    calendar_days: int = 4
    atr_window: int = 14
    trail_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RevivalParams]):
    strategy_id = 'gen_local-w2_1791332434'

    @classmethod
    def params_type(cls) -> type[RevivalParams]:
        return RevivalParams

    @staticmethod
    def warmup_bars(params: RevivalParams) -> int:
        return int(max(params.memory_window + params.revival_window + 3,
                       params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RevivalParams) -> pd.DataFrame:
        if (params.memory_window < 3 or params.revival_window < 1
                or not 1 <= params.min_revivals <= params.revival_window
                or not 1 <= params.calendar_days <= 7
                or params.atr_window < 1
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0
                or not 5 <= params.hold_bars <= 10):
            raise ValueError('Invalid strategy parameters')

        close = data['close'].astype(float)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = returns.shift(2), returns.shift(1), returns
        valid = (a.notna() & b.notna() & c.notna()
                 & a.ne(b) & a.ne(c) & b.ne(c))
        # Three pairwise comparisons encode all six strict permutations.
        ordinal = (4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int)
                   + b.gt(c).astype(int)).astype(float).where(valid)
        history_ready = (valid.astype(int)
                         .rolling(params.memory_window, min_periods=params.memory_window)
                         .sum().shift(1).eq(params.memory_window))
        revived = pd.Series(False, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            occurrence = ordinal.eq(code)
            previous_count = (occurrence.astype(float)
                              .rolling(params.memory_window, min_periods=params.memory_window)
                              .sum().shift(1))
            revived |= occurrence & previous_count.eq(0)
        novelty = revived.astype(float).where(valid & history_ready)
        revival_count = novelty.rolling(
            params.revival_window, min_periods=params.revival_window
        ).sum()

        day = data.index.day
        month_length = data.index.days_in_month
        seasonal = pd.Series(
            (day <= params.calendar_days)
            | (day > month_length - params.calendar_days),
            index=data.index,
            dtype=bool,
        )
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False).replace([np.inf, -np.inf], np.nan)
        atr = true_range.rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return pd.DataFrame({
            'return': returns,
            'ordinal_code': ordinal,
            'history_ready': history_ready,
            'pattern_revival': novelty,
            'revival_count': revival_count,
            'seasonal_window': seasonal,
            'true_range': true_range,
            'atr': atr,
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RevivalParams,
    ) -> SignalFrame:
        entry = (indicators['revival_count'].ge(params.min_revivals)
                 & indicators['seasonal_window'].fillna(False).astype(bool))
        prices = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = entry.fillna(False).to_numpy(dtype=bool)
        targets = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        peak = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            price, atr = prices[i], atrs[i]
            usable = np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if usable:
                    # Entry fills on this bar after yesterday's decision.
                    peak = max(peak, price)
                    stop = max(stop, peak - params.trail_atr * atr)
                if (usable and price <= stop) or age >= params.hold_bars:
                    active = False
            elif entries[i] and usable:
                active = True
                age = 0
                # Start tracking closes on the first actual holding bar.
                peak = -np.inf
                stop = -np.inf
            targets[i] = int(active)

        df = pd.DataFrame({'signal': targets, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # ATR expansion must not lower an established stop; exits fill next bar.
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({'close': [100., 102., 105., 104., 102., 101.]}, index=index)
    derived = pd.DataFrame({
        'revival_count': [1., 0., 0., 0., 0., 0.],
        'seasonal_window': True,
        'atr': [1., 1., 1., 5., 5., 5.],
    }, index=index)
    params = RevivalParams(trail_atr=2.0)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    derived['revival_count'] = 0.0
    assert GeneratedStrategy.generate_signals(data, derived, None, params).data['signal'].eq(0).all()
    derived['revival_count'] = 1.0
    derived['seasonal_window'] = False
    assert GeneratedStrategy.generate_signals(data, derived, None, params).data['signal'].eq(0).all()
