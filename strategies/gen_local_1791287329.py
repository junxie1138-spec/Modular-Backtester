from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    frac_window: int = 32
    frac_order: float = 0.45
    percentile_window: int = 126
    tail_percentile: float = 0.15
    excursion_window: int = 63
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    initial_stop_atr: float = 3.0
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.frac_window, self.percentile_window, self.excursion_window, self.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError('frac_order must lie between zero and one.')
        if not 0.0 < self.tail_percentile < 0.5:
            raise ValueError('tail_percentile must lie between zero and one half.')
        if min(self.breakeven_pct, self.trail_atr, self.initial_stop_atr) <= 0 or self.holding_bars < 1:
            raise ValueError('Exit parameters must be positive.')


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791287329'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params):
        return int(max(params.frac_window, params.excursion_window, params.atr_window + 1) + params.percentile_window + 1)

    def indicators(self, data, params):
        close = data['close'].astype(float).where(data['close'] > 0)
        out = pd.DataFrame(index=data.index)
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        fractional = close * weights[0]
        for k in range(1, params.frac_window):
            fractional = fractional + weights[k] * close.shift(k)
        # Remove finite-filter constant leakage using the oldest observed close.
        out['fractional_close'] = fractional
        out['fractional_displacement'] = (fractional - weights.sum() * close.shift(params.frac_window - 1)) / close
        history = out['fractional_displacement'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window)
        out['lower_percentile'] = history.quantile(params.tail_percentile)
        out['upper_percentile'] = history.quantile(1.0 - params.tail_percentile)
        peak = close.rolling(params.excursion_window, min_periods=params.excursion_window).max()
        trough = close.rolling(params.excursion_window, min_periods=params.excursion_window).min()
        out['drawdown'] = close / peak - 1.0
        out['drawup'] = close / trough - 1.0
        out['drawdown_percentile'] = out['drawdown'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(0.5)
        out['drawup_percentile'] = out['drawup'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(0.5)
        previous = close.shift(1)
        true_range = pd.concat([data['high'] - data['low'], (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1, skipna=False)
        out['atr'] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        fd = out['fractional_displacement']
        out['long_entry'] = ((fd.shift(1) <= out['lower_percentile'].shift(1)) & (fd > out['lower_percentile']) & (out['drawdown'] < out['drawdown_percentile']) & (close > previous))
        out['short_entry'] = ((fd.shift(1) >= out['upper_percentile'].shift(1)) & (fd < out['upper_percentile']) & (out['drawup'] > out['drawup_percentile']) & (close < previous))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        count = len(data)
        signals = np.zeros(count, dtype=int)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        ready = indicators[['fractional_displacement', 'lower_percentile', 'upper_percentile', 'drawdown_percentile', 'drawup_percentile', 'atr']].notna().all(axis=1).to_numpy()
        side = 0
        entry = np.nan
        entry_bar = -1
        protected_profit = -np.inf
        best_profit = 0.0
        activated = False
        warmup = self.warmup_bars(params)
        for i in range(count):
            if side != 0:
                if np.isnan(entry):
                    # Yesterday's decision fills at today's open.
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        side = 0
                        continue
                    entry_bar = i
                    protected_profit = -params.initial_stop_atr * atr[i - 1]
                    best_profit = 0.0
                    activated = False
                if not np.isfinite(closes[i]) or closes[i] <= 0:
                    signals[i] = side
                    continue
                profit = side * (closes[i] - entry)
                # Stops are evaluated at closes and executed next bar.
                if profit <= protected_profit or i - entry_bar + 1 >= params.holding_bars:
                    side = 0
                    entry = np.nan
                    continue
                best_profit = max(best_profit, profit)
                if profit >= entry * params.breakeven_pct:
                    activated = True
                if activated:
                    protected_profit = max(protected_profit, 0.0)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        protected_profit = max(protected_profit, best_profit - params.trail_atr * atr[i])
                    if profit <= protected_profit:
                        side = 0
                        entry = np.nan
                        continue
                # Protected profit only rises: long stops rise; short stops fall.
                signals[i] = side
            elif i >= warmup and ready[i] and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                if longs[i]:
                    side = 1
                elif shorts[i]:
                    side = -1
                if side:
                    entry = np.nan
                    signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': np.full(count, 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RecoveryParams()
    strategy = object.__new__(GeneratedStrategy)
    rng = np.random.default_rng(17)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 900)))
    opening = np.r_[close[0], close[:-1]]
    data = pd.DataFrame({'open': opening, 'high': np.maximum(opening, close) * 1.005, 'low': np.minimum(opening, close) * 0.995, 'close': close, 'volume': np.full(900, 1000000.0)}, index=pd.date_range('2020-01-01', periods=900, freq='B'))
    full = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    prefix = data.iloc[:700]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    assert full.index.equals(data.index)
    assert full['signal'].isin([-1, 0, 1]).all() and (full['size'] > 0).all()
    assert full['signal'].iloc[:strategy.warmup_bars(params) + 1].eq(0).all()
    assert np.array_equal(full['signal'].iloc[:700].to_numpy(), partial['signal'].to_numpy())
