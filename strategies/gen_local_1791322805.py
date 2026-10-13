from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    rank_window: int = 40
    boundary_smoothing: int = 3
    entry_percentile: float = 0.60
    minimum_rank_gap: float = 0.15
    atr_window: int = 14
    initial_stop_atr: float = 2.0
    breakeven_gain: float = 0.01
    trail_atr: float = 2.5
    max_hold: int = 10

    def __post_init__(self):
        for value in (self.rank_window, self.boundary_smoothing, self.atr_window, self.max_hold):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Lookbacks and max_hold must be positive integers')
        if self.rank_window < 2:
            raise ValueError('rank_window must be at least two')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must lie between zero and one')
        if not 0.0 < self.minimum_rank_gap < 1.0:
            raise ValueError('minimum_rank_gap must lie between zero and one')
        for value in (self.initial_stop_atr, self.breakeven_gain, self.trail_atr):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError('Stop parameters must be finite and positive')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1791322805'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        # Include smoothing, the subsequent rank window, and crossover history.
        return int(max(params.rank_window + params.boundary_smoothing, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['smoothed_low'] = data['low'].rolling(params.boundary_smoothing).mean()
        out['smoothed_high'] = data['high'].rolling(params.boundary_smoothing).mean()
        out['close_rank'] = data['close'].rolling(params.rank_window).rank(pct=True)
        out['low_rank'] = out['smoothed_low'].rolling(params.rank_window).rank(pct=True)
        out['high_rank'] = out['smoothed_high'].rolling(params.rank_window).rank(pct=True)
        out['rank_gap'] = out['low_rank'] - out['high_rank']
        out['previous_close_rank'] = out['close_rank'].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        # Primitive one: an upward crossing of the closing-price percentile.
        crossing = (
            indicators['close_rank'].gt(params.entry_percentile)
            & indicators['previous_close_rank'].le(params.entry_percentile)
        )
        # Primitive two: historical low ranks lead historical high ranks.
        floor_leads = indicators['rank_gap'].ge(params.minimum_rank_gap)
        entries = (crossing & floor_leads & indicators['atr'].gt(0.0)).fillna(False).to_numpy(dtype=bool)

        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry = stop = peak = 0.0
        held = 0
        warmup = self.warmup_bars(params)

        for i in range(len(data)):
            if pending:
                pending = False
                # Record the next observed open only when that bar arrives.
                # ponytail: nominal open is the entry reference; use fill feedback if the contract adds it.
                if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                    continue
                entry = float(opens[i])
                peak = entry
                stop = entry - params.initial_stop_atr * atr[i - 1]
                held = 0
                armed = False
                active = True

            if active:
                held += 1
                if not np.isfinite([highs[i], lows[i], closes[i]]).all() or closes[i] <= 0.0:
                    active = False
                    continue

                prior_stop = stop
                peak = max(peak, float(highs[i]))
                if peak >= entry * (1.0 + params.breakeven_gain):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr[i]) and atr[i] > 0.0:
                        stop = max(stop, peak - params.trail_atr * atr[i])

                # Test the old stop against the low. A newly raised stop is
                # tested against the close, avoiding assumed high/low ordering.
                exit_fired = lows[i] <= prior_stop or closes[i] <= stop
                if exit_fired or held >= params.max_hold:
                    active = False
                else:
                    raw[i] = 1
            elif i + 1 >= warmup and entries[i]:
                pending = True
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GeneratedParams(rank_window=2, boundary_smoothing=1, atr_window=2)
    index = pd.date_range('2020-01-01', periods=9, freq='B')
    data = pd.DataFrame({
        'open': [100, 100, 100, 100, 100, 102, 103, 102, 101],
        'high': [100.5, 100.5, 100.5, 100.5, 100.5, 104, 104, 102.5, 101.5],
        'low': [99.5, 99.5, 99.5, 99.5, 99.5, 101, 102.5, 101, 100.5],
        'close': [100, 100, 100, 100, 100, 103, 103, 101.25, 101],
        'volume': [1000] * 9,
    }, index=index)
    indicators = pd.DataFrame({
        'close_rank': [np.nan, np.nan, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        'previous_close_rank': [np.nan, np.nan, np.nan, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8],
        'rank_gap': [0.25] * 9,
        'atr': [1, 1, 1, 1, 1, 1, 10, 10, 10],
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    signal = strategy.generate_signals(data, indicators, ctx, params).data['signal']
    # ATR expands after activation: the stop must remain at 101.5.
    assert signal.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0]
    blocked = indicators.copy()
    blocked['rank_gap'] = 0.0
    assert strategy.generate_signals(data, blocked, ctx, params).data['signal'].eq(0).all()
    params.trail_atr = 20.0
    breakeven_data = data.copy()
    breakeven_data.loc[index[7], ['low', 'close']] = [99.75, 99.8]
    assert strategy.generate_signals(breakeven_data, indicators, ctx, params).data['signal'].iloc[8] == 0
    params.max_hold = 2
    assert strategy.generate_signals(data, indicators, ctx, params).data['signal'].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0]
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert actual['close_rank'].iloc[:params.rank_window - 1].isna().all()
    assert strategy.generate_signals(data, actual, ctx, params).data['signal'].isin([0, 1]).all()


if __name__ == '__main__':
    _self_check()
