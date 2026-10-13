from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StaircaseParams:
    streak_min: int = 3
    progress_bars: int = 5
    progress_atr: float = 1.2
    settlement_min: float = 0.6
    atr_bars: int = 14
    initial_stop_atr: float = 2.0
    breakeven_pct: float = 0.02
    trail_atr: float = 2.0
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StaircaseParams]):
    strategy_id = 'gen_local_1791270909'

    @classmethod
    def params_type(cls) -> type[StaircaseParams]:
        return StaircaseParams

    def warmup_bars(self, params: StaircaseParams) -> int:
        return int(max(params.atr_bars + 1, params.progress_bars + 1, params.streak_min + 2))

    def indicators(self, data: pd.DataFrame, params: StaircaseParams) -> pd.DataFrame:
        if min(params.streak_min, params.progress_bars, params.atr_bars, params.max_hold_bars) < 1:
            raise ValueError('Lookbacks and holding duration must be positive.')
        values = [params.progress_atr, params.settlement_min, params.initial_stop_atr, params.breakeven_pct, params.trail_atr]
        if not np.isfinite(values).all() or min(params.progress_atr, params.initial_stop_atr, params.breakeven_pct, params.trail_atr) <= 0 or not 0 <= params.settlement_min <= 1:
            raise ValueError('Invalid thresholds or stop parameters.')
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['higher_low'] = data['low'].gt(data['low'].shift(1)).fillna(False)
        groups = (~out['higher_low']).cumsum()
        out['streak'] = out['higher_low'].astype(int).groupby(groups).cumsum()
        spread = (data['high'] - data['low']).where(lambda x: x > 0)
        out['settlement'] = (data['close'] - data['low']) / spread
        out['progress'] = data['close'].diff(params.progress_bars) / out['atr'].where(out['atr'] > 0)
        out['qualified'] = (
            out['streak'].ge(params.streak_min)
            & out['settlement'].ge(params.settlement_min)
            & out['progress'].ge(params.progress_atr)
            & np.isfinite(out['atr'])
            & out['atr'].gt(0)
        ).fillna(False)
        out['confirmed'] = (
            out['qualified']
            & out['qualified'].shift(1, fill_value=False)
            & (np.arange(len(data)) >= self.warmup_bars(params))
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StaircaseParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        holding = False
        pending_entry = False
        armed = False
        entry = 0.0
        stop = 0.0
        entry_atr = 0.0
        held_bars = 0
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if pending_entry:
                pending_entry = False
                entry = opens[i]
                if not np.isfinite(entry) or entry <= 0:
                    holding = False
                    continue
                stop = max(0.0, entry - params.initial_stop_atr * entry_atr)
                armed = False
                held_bars = 0
            if holding:
                held_bars += 1
                if not np.isfinite(close) or close <= 0 or close <= stop or held_bars >= params.max_hold_bars:
                    holding = False
                    continue
                if close >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, close - params.trail_atr * atr)
                signals[i] = 1
            elif confirmed[i] and np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0:
                holding = True
                pending_entry = True
                entry_atr = atr
                signals[i] = 1
        # Stops are evaluated at the close; exits execute at the next bar.
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({
        'open': 100.0,
        'high': [101.0, 101.0, 101.0, 104.0, 105.0, 102.0, 101.0],
        'low': 99.0,
        'close': [100.0, 100.0, 100.0, 103.0, 104.0, 101.0, 100.0],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'confirmed': [False, False, True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, StaircaseParams()).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
    rising = data.copy()
    rising['low'] = np.arange(7, dtype=float) + 99.0
    rising['high'] = rising['low'] + 1.0
    rising['close'] = rising['low'] + 0.9
    rising['open'] = rising['low'] + 0.2
    params = StaircaseParams(streak_min=1, progress_bars=1, progress_atr=0.1, atr_bars=1)
    derived = strategy.indicators(rising, params)
    assert derived['streak'].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert not derived['confirmed'].iloc[0]
    assert not (derived['confirmed'] & ~derived['qualified'].shift(1, fill_value=False)).any()


if __name__ == '__main__':
    _self_check()
