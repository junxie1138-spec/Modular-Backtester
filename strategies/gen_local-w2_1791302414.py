from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    baseline: int = 40
    energy_window: int = 3
    change_threshold: float = 0.25
    energy_ratio: float = 1.5
    atr_window: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791302414'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(8 + params.energy_window + params.baseline + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.baseline, params.energy_window, params.atr_window) < 1:
            raise ValueError('Lookbacks must be positive')
        if not 3 <= params.hold_days <= 5:
            raise ValueError('hold_days must be between 3 and 5')
        if not (0 < params.change_threshold < 1 and params.energy_ratio > 0
                and params.breakeven_pct > 0 and params.trail_atr > 0):
            raise ValueError('Invalid thresholds')
        out = pd.DataFrame(index=data.index)
        out['log_price'] = np.log(data['close'].where(data['close'] > 0))
        energies = []
        for half in (1, 2, 4):
            mean = out['log_price'].rolling(half, min_periods=half).mean()
            out[f'mean_{half}'] = mean
            coefficient = (mean - mean.shift(half)) * np.sqrt(half / 2.0)
            out[f'haar_{half}'] = coefficient
            name = f'energy_{half}'
            out[name] = coefficient.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window).mean()
            energies.append(name)
        out['total_energy'] = out[energies].sum(axis=1, min_count=3)
        denominator = out['total_energy'].where(out['total_energy'] > 0)
        distances = []
        for half in (1, 2, 4):
            out[f'share_{half}'] = out[f'energy_{half}'] / denominator
            out[f'baseline_{half}'] = out[f'share_{half}'].shift(1).rolling(
                params.baseline, min_periods=params.baseline).mean()
            name = f'distance_{half}'
            out[name] = (out[f'share_{half}'] - out[f'baseline_{half}']).abs()
            distances.append(name)
        out['change'] = 0.5 * out[distances].sum(axis=1, min_count=3)
        out['baseline_energy'] = out['total_energy'].shift(1).rolling(
            params.baseline, min_periods=params.baseline).mean()
        out['energy_ratio'] = out['total_energy'] / out['baseline_energy'].where(
            out['baseline_energy'] > 0)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out['direction'] = np.sign(out['haar_4']).fillna(0).astype(int)
        out['body'] = data['close'] - data['open']
        out['event'] = (
            (out['change'] >= params.change_threshold)
            & (out['change'].shift(1) < params.change_threshold)
            & (out['share_4'] > out['baseline_4'])
            & (out['share_1'] < out['baseline_1'])
            & (out['energy_ratio'] >= params.energy_ratio)
            & (out['body'] * out['direction'] > 0)
            & (out['atr'] > 0)
        ).fillna(False)
        out['strength'] = (0.25 + 0.70 * (
            (out['change'] - params.change_threshold)
            / params.change_threshold).clip(0, 1)).fillna(0.25)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({'signal': 0, 'size': 0.25}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        position = 0
        entry = np.nan
        best = -np.inf
        stop = -np.inf
        armed = False
        age = 0
        size = 0.25
        for i in range(len(data)):
            row = indicators.iloc[i]
            close = float(data['close'].iloc[i])
            atr = float(row['atr'])
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if position:
                # The previous close's decision enters at this bar's open.
                if np.isnan(entry):
                    entry = float(data['open'].iloc[i])
                    if not np.isfinite(entry) or entry <= 0:
                        position = 0
                    else:
                        best = position * entry
                if position:
                    age += 1
                    if not valid:
                        position = 0
                    else:
                        # Signed price makes both long and short stops ratchet upward.
                        price = position * close
                        stopped = armed and price <= stop
                        best = max(best, price)
                        if best - position * entry >= entry * params.breakeven_pct:
                            armed = True
                        if armed:
                            stop = max(stop, position * entry,
                                       best - params.trail_atr * atr)
                            stopped = stopped or price <= stop
                        if stopped or age >= params.hold_days:
                            position = 0
                # Exits are close decisions, executed next open; no same-close reentry.
            elif valid and pd.notna(row['event']) and bool(row['event']):
                direction = int(row['direction'])
                if direction in (-1, 1):
                    position = direction
                    entry = np.nan
                    best = stop = -np.inf
                    armed = False
                    age = 0
                    strength = float(row['strength'])
                    size = float(np.clip(strength, 0.25, 0.95)) if np.isfinite(strength) else 0.25
            signals[i] = position
            sizes[i] = size if position else 0.25
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=7)
    for direction in (1, -1):
        close = np.array([100, 100, 102, 100.5, 100, 100, 100], dtype=float)
        if direction == -1:
            close = 200 - close
        data = pd.DataFrame({'open': 100.0, 'high': close + 3,
                             'low': close - 3, 'close': close,
                             'volume': 1000.0}, index=index)
        indicators = pd.DataFrame({'atr': 0.5, 'event': False,
                                   'direction': direction, 'strength': 0.8}, index=index)
        indicators.loc[index[0], 'event'] = True
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, 0, 0, 0]
        assert result['size'].iloc[1] == 0.8
        for end in range(1, len(data) + 1):
            prefix = strategy.generate_signals(data.iloc[:end], indicators.iloc[:end], None, params).data
            pd.testing.assert_frame_equal(prefix, result.iloc[:end])
        data['close'] = 100.0
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, direction, 0, 0]


if __name__ == '__main__':
    _self_check()
