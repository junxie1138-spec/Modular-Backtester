from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    extreme_window: int = 20
    percentile_window: int = 63
    percentile: float = 0.8
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791339900'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.extreme_window + params.percentile_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.extreme_window, params.percentile_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars')
        if not 0.5 < params.percentile < 1.0:
            raise ValueError('Percentile must be between 0.5 and 1.0')
        if not np.isfinite(params.breakeven_pct) or params.breakeven_pct <= 0:
            raise ValueError('Breakeven percentage must be finite and positive')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('ATR multiple must be finite and positive')
        if params.hold_bars not in (1, 2):
            raise ValueError('Holding horizon must be one or two sessions')
        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1)
        out['peak'] = out['prior_close'].rolling(params.extreme_window).max()
        out['trough'] = out['prior_close'].rolling(params.extreme_window).min()
        out['drawdown'] = out['peak'] / out['prior_close'] - 1.0
        out['drawup'] = out['prior_close'] / out['trough'] - 1.0
        out['gap'] = data['open'] / out['prior_close'] - 1.0
        out['long_reclamation'] = (out['gap'].clip(lower=0) / out['drawdown'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out['short_reclamation'] = ((-out['gap']).clip(lower=0) / out['drawup'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for column in ('drawdown', 'drawup', 'long_reclamation', 'short_reclamation'):
            out[column + '_threshold'] = out[column].rolling(params.percentile_window).quantile(params.percentile).shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['prior_close']).abs(),
            (data['low'] - out['prior_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        long_entry = (
            (out['gap'] > 0)
            & (out['drawdown'] > 0)
            & (out['drawdown'] >= out['drawdown_threshold'])
            & (out['long_reclamation'] > out['long_reclamation_threshold'])
            & (data['close'] >= data['open'])
            & (data['close'] < out['peak'])
        )
        short_entry = (
            (out['gap'] < 0)
            & (out['drawup'] > 0)
            & (out['drawup'] >= out['drawup_threshold'])
            & (out['short_reclamation'] > out['short_reclamation_threshold'])
            & (data['close'] <= data['open'])
            & (data['close'] > out['trough'])
        )
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out.loc[~np.isfinite(out['atr']) | (out['atr'] <= 0), 'entry'] = 0
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        df = pd.DataFrame({'signal': np.zeros(len(data), dtype=int), 'size': 1.0}, index=data.index)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = pending = age = 0
        entry = stop = best = 0.0
        armed = False
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            if pending:
                position, pending = pending, 0
                entry = opens[i]
                best = position * entry
                stop = best - params.trail_atr * atr[i - 1]
                age, armed = 0, False
            if position:
                age += 1
                signed_close = position * closes[i]
                adverse = lows[i] if position == 1 else -highs[i]
                # Test the existing stop before updating it at this close.
                exit_now = adverse <= stop
                if not exit_now and np.isfinite(atr[i]) and atr[i] > 0:
                    best = max(best, signed_close)
                    if best - position * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed-price stops only rise; short price stops only fall.
                        stop = max(stop, position * entry, best - params.trail_atr * atr[i])
                        exit_now = signed_close <= stop
                elif not np.isfinite(atr[i]) or atr[i] <= 0:
                    exit_now = True
                # Breaches detected during this bar execute next open.
                if exit_now or age >= params.hold_bars:
                    position = 0
                decisions[i] = position
            elif i >= warmup - 1 and entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = int(entries[i])
                decisions[i] = pending
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = RecoveryParams(extreme_window=2, percentile_window=2, atr_window=2)
    for side in (-1, 1):
        for stopped in (False, True):
            index = pd.date_range('2020-01-01', periods=9)
            data = pd.DataFrame({'open': 100.0, 'high': 100.1, 'low': 99.9, 'close': 100.0, 'volume': 1000.0}, index=index)
            data.loc[index[4], 'close'] = 100.0 + side * 5.0
            data.loc[index[5], 'close'] = 100.0 + side * (-2.0 if stopped else 1.0)
            data['high'] = data[['open', 'close']].max(axis=1) + 0.1
            data['low'] = data[['open', 'close']].min(axis=1) - 0.1
            indicators = strategy.indicators(data, params)
            indicators['entry'] = 0
            indicators.loc[index[4], 'entry'] = side
            indicators['atr'] = 1.0
            ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
            result = strategy.generate_signals(data, indicators, ctx, params).data
            expected = [0, 0, 0, 0, 0, side, 0 if stopped else side, 0, 0]
            assert result['signal'].tolist() == expected
            assert (result['size'] > 0).all()
