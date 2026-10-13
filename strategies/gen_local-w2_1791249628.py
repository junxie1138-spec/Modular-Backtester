from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    channel_window: int = 20
    energy_window: int = 6
    atr_window: int = 14
    fine_threshold: float = 0.50
    boundary: float = 0.20
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791249628'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.channel_window, params.atr_window + 1,
                       8 + params.energy_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (params.channel_window < 2 or params.energy_window < 2
                or params.atr_window < 1 or params.hold_bars < 1
                or not 0 < params.boundary < 0.5
                or not 0 < params.fine_threshold < 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['log_close'] = np.log(close.where(close > 0))
        out['return'] = out['log_close'].diff()
        # Causal Haar details compare adjacent, non-overlapping return blocks.
        for half in (1, 2, 4):
            out[f'mean_{half}'] = out['return'].rolling(half).mean()
            out[f'detail_{half}'] = np.sqrt(half / 2.0) * (
                out[f'mean_{half}'] - out[f'mean_{half}'].shift(half))
            out[f'energy_{half}'] = out[f'detail_{half}'].pow(2).rolling(
                params.energy_window).mean()
        out['total_energy'] = out['energy_1'] + out['energy_2'] + out['energy_4']
        out['fine_share'] = out['energy_1'] / out['total_energy'].where(
            out['total_energy'] > 0)
        out['previous_fine_energy'] = out['energy_1'].shift(1)
        out['channel_low'] = close.rolling(params.channel_window).min()
        out['channel_high'] = close.rolling(params.channel_window).max()
        out['channel_width'] = out['channel_high'] - out['channel_low']
        out['position'] = (close - out['channel_low']) / out['channel_width'].where(
            out['channel_width'] > 0)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = out[['fine_share', 'position', 'atr', 'energy_1',
                            'previous_fine_energy']].notna().all(axis=1) & (out['atr'] > 0)
        out['release'] = (out['ready']
                          & (out['fine_share'] >= params.fine_threshold)
                          & (out['energy_1'] < out['previous_fine_energy']))
        out['candidate'] = np.select([
            out['release'] & (out['position'] <= params.boundary),
            out['release'] & (out['position'] >= 1.0 - params.boundary)
        ], [1, -1], default=0).astype(int)
        out['strength'] = (2.0 * (out['position'] - 0.5).abs()
                           * out['fine_share']).clip(0.0, 1.0).fillna(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.15, dtype=float)
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        candidates = indicators['candidate'].to_numpy(dtype=int)
        atr = indicators['atr'].to_numpy(dtype=float)
        position = indicators['position'].to_numpy(dtype=float)
        strength = indicators['strength'].to_numpy(dtype=float)
        ready = indicators['ready'].to_numpy(dtype=bool)
        side = 0
        entry = np.nan
        frozen_atr = 0.0
        age = 0
        trade_size = 0.15
        for i in range(n):
            if side:
                # The previous close's entry decision fills at this open.
                if np.isnan(entry):
                    entry = opens[i]
                age += 1
                valid = (ready[i] and np.isfinite(close[i])
                         and np.isfinite(entry) and entry > 0)
                stopped = valid and side * (close[i] - entry) < -params.stop_atr * frozen_atr
                centered = valid and side * (position[i] - 0.5) >= 0
                if not valid or stopped or centered or age >= params.hold_bars:
                    side = 0
                    entry = np.nan
                    trade_size = 0.15
            elif (ready[i] and candidates[i] != 0
                  and np.isfinite(close[i]) and close[i] > 0
                  and np.isfinite(atr[i]) and atr[i] > 0):
                side = int(candidates[i])
                entry = np.nan
                frozen_atr = float(atr[i])
                age = 0
                trade_size = 0.15 + 0.80 * float(np.clip(strength[i], 0.0, 1.0))
            signals[i] = side
            sizes[i] = trade_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    # Check delayed entries, actual opening-price stops, both sides, and sizing.
    index = pd.date_range('2020-01-01', periods=5)
    bars = pd.DataFrame({'open': [100., 110., 100., 90., 100.],
                         'close': [100., 107., 100., 93., 100.]}, index=index)
    features = pd.DataFrame({'candidate': [1, 0, -1, 0, 0],
                             'atr': 1.0,
                             'position': [0.1, 0.1, 0.9, 0.9, 0.5],
                             'strength': [0.5, 0.0, 0.8, 0.0, 0.0],
                             'ready': True}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(bars, features, None, Params()).data
    assert result['signal'].tolist() == [0, 1, 0, -1, 0]
    assert np.allclose(result['size'].iloc[[1, 3]], [0.55, 0.79])
    assert (result['size'] > 0).all()
