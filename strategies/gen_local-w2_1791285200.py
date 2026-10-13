from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CycleParams:
    signature_window: int = 12
    correlation_window: int = 30
    phase_window: int = 3
    area_threshold: float = 0.12
    correlation_floor: float = 0.05
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[CycleParams]):
    strategy_id = 'gen_local-w2_1791285200'

    @classmethod
    def params_type(cls) -> type[CycleParams]:
        return CycleParams

    def warmup_bars(self, params: CycleParams) -> int:
        return int(max(params.signature_window + 2, params.correlation_window + 2, params.phase_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: CycleParams) -> pd.DataFrame:
        if min(params.signature_window, params.correlation_window, params.phase_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if params.phase_window >= params.signature_window:
            raise ValueError('phase_window must be shorter than signature_window.')
        if params.max_hold < 1 or params.trail_atr <= 0 or params.breakeven_pct <= 0 or params.area_threshold < 0:
            raise ValueError('Invalid exit or area parameters.')
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['lagged_return'] = out['return'].shift(1)
        r = out['return']
        length = params.signature_window
        # Second-order antisymmetric signature of the return/lagged-return path.
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, length):
            cross = r.shift(lag) * r.shift(1) - r.shift(lag + 1) * r
            area = area + cross.rolling(length - lag, min_periods=length - lag).sum()
        out['signature_area'] = 0.5 * area
        out['signature_energy'] = 0.5 * (r.pow(2) + out['lagged_return'].pow(2)).rolling(length, min_periods=length).sum()
        out['normalized_area'] = out['signature_area'].div(out['signature_energy'].where(out['signature_energy'] > 0))
        out['autocorrelation'] = r.rolling(params.correlation_window, min_periods=params.correlation_window).corr(out['lagged_return'])
        out['fast_return'] = r.rolling(params.phase_window, min_periods=params.phase_window).mean()
        out['slow_return'] = r.rolling(length, min_periods=length).mean()
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[['normalized_area', 'autocorrelation', 'fast_return', 'slow_return', 'atr']].notna().all(axis=1) & out['atr'].gt(0)
        cycle = ready & out['normalized_area'].gt(params.area_threshold) & out['autocorrelation'].gt(params.correlation_floor)
        buy = cycle & out['slow_return'].lt(0) & out['fast_return'].gt(out['slow_return'])
        sell = cycle & out['slow_return'].gt(0) & out['fast_return'].lt(out['slow_return'])
        out['candidate'] = np.select([buy, sell], [1, -1], default=0).astype(int)
        out['confirmed'] = out['candidate'].where(out['candidate'].eq(out['candidate'].shift(1)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CycleParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].to_numpy(dtype=int)
        side = 0
        pending = False
        entry = 0.0
        entry_bar = 0
        stop = -np.inf
        extreme = -np.inf
        activated = False
        for i in range(len(data)):
            if pending:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError('Entry open must be finite and positive.')
                entry = opens[i]
                entry_bar = i
                extreme = side * entry
                stop = extreme - params.trail_atr * atrs[i - 1]
                activated = False
                pending = False
            if side:
                close = closes[i]
                atr = atrs[i]
                if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                    side = 0
                else:
                    # Favorable-price coordinates make the stop increase for both directions.
                    # The corresponding short-price stop decreases as profits grow.
                    favorable = side * close
                    extreme = max(extreme, favorable)
                    activated = activated or side * (close - entry) / entry >= params.breakeven_pct
                    if activated:
                        stop = max(stop, side * entry, extreme - params.trail_atr * atr)
                    # Stops are close-observed decisions, executed at the next open.
                    if favorable <= stop or i - entry_bar + 1 >= params.max_hold:
                        side = 0
                signals[i] = side
                continue
            if confirmed[i] and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(confirmed[i])
                pending = True
            signals[i] = side
        df['signal'] = signals
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = CycleParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=7)
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0, 0, 1, 3, 1, 0, 0], dtype=float)
        sample = pd.DataFrame({'open': 100.0, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000.0}, index=index)
        derived = pd.DataFrame({'atr': 1.0, 'confirmed': [0, direction, direction, direction, direction, direction, 0]}, index=index)
        result = strategy.generate_signals(sample, derived, None, params).data
        assert result['signal'].tolist() == [0, 0, direction, direction, direction, direction, 0]
        assert result['size'].gt(0).all()
    flat = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=pd.date_range('2020-01-01', periods=80))
    derived = strategy.indicators(flat, params)
    assert derived['confirmed'].eq(0).all()
    assert strategy.generate_signals(flat, derived, None, params).data['signal'].eq(0).all()
