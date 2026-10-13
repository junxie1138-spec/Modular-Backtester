from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ResonanceParams:
    energy_window: int = 5
    baseline_window: int = 20
    atr_window: int = 14
    resonance_share: float = 0.50
    energy_multiple: float = 1.10
    coherence_min: float = 0.65
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ResonanceParams]):
    strategy_id = 'gen_local-w2_1791206765'

    @classmethod
    def params_type(cls):
        return ResonanceParams

    @staticmethod
    def warmup_bars(params: ResonanceParams) -> int:
        return int(max(16 + params.energy_window + params.baseline_window, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ResonanceParams) -> pd.DataFrame:
        if min(params.energy_window, params.baseline_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError('Window lengths and holding period must be positive.')
        if not (0 < params.resonance_share < 1 and 0 < params.coherence_min <= 1):
            raise ValueError('Energy share and coherence thresholds must be valid fractions.')
        if not (np.isfinite(params.stop_atr) and params.stop_atr > 0 and np.isfinite(params.energy_multiple) and params.energy_multiple > 0):
            raise ValueError('ATR multiplier and energy multiplier must be positive and finite.')
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        for half in (2, 4, 8):
            out[f'mean_{half}'] = out['log_close'].rolling(half, min_periods=half).mean()
            out[f'haar_{half}'] = (out[f'mean_{half}'] - out[f'mean_{half}'].shift(half)) * np.sqrt(half / 2.0)
            out[f'energy_{half}'] = out[f'haar_{half}'].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
            out[f'signed_energy_{half}'] = (np.sign(out[f'haar_{half}']) * out[f'haar_{half}'].pow(2)).rolling(params.energy_window, min_periods=params.energy_window).mean()
        energy_cols = ['energy_2', 'energy_4', 'energy_8']
        signed_cols = ['signed_energy_2', 'signed_energy_4', 'signed_energy_8']
        out['total_energy'] = out[energy_cols].sum(axis=1, min_count=3)
        denominator = out['total_energy'].where(out['total_energy'] > 0)
        out['middle_share'] = out['energy_4'] / denominator
        out['coherence'] = out[signed_cols].sum(axis=1, min_count=3) / denominator
        out['baseline_energy'] = out['total_energy'].shift(1).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out['energy_ratio'] = out['total_energy'] / out['baseline_energy'].where(out['baseline_energy'] > 0)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        ready = np.isfinite(out[['middle_share', 'coherence', 'energy_ratio', 'atr']]).all(axis=1) & out['atr'].gt(0)
        resonant = ready & out['middle_share'].ge(params.resonance_share) & out['energy_ratio'].ge(params.energy_multiple)
        bullish = resonant & out['coherence'].ge(params.coherence_min) & out['haar_4'].gt(0) & out['haar_8'].gt(0)
        bearish = resonant & out['coherence'].le(-params.coherence_min) & out['haar_4'].lt(0) & out['haar_8'].lt(0)
        out['candidate'] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ResonanceParams) -> SignalFrame:
        candidate = indicators['candidate'].fillna(0).astype(int)
        confirmed = candidate.where(candidate.ne(0) & candidate.eq(candidate.shift(1)), 0).to_numpy()
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_atr = np.nan
        stop_price = np.nan
        held = 0
        for i in range(len(data)):
            if position:
                # The preceding close's order fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    stop_price = entry_price - position * params.stop_atr * entry_atr
                held += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                stopped = valid and ((position == 1 and closes[i] < stop_price) or (position == -1 and closes[i] > stop_price))
                if not valid or stopped or held >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
                    raw[i] = 0
                    continue
            elif confirmed[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(confirmed[i])
                entry_price = np.nan
                entry_atr = atr[i]
                held = 0
            raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({'open': [100.0] * 9, 'high': [103.0] * 9, 'low': [97.0] * 9, 'close': [100.0, 100.0, 100.0, 97.0, 100.0, 100.0, 103.0, 100.0, 100.0], 'volume': [1000.0] * 9}, index=index)
    ind = pd.DataFrame({'candidate': [1, 1, 0, 0, -1, -1, 0, 0, 0], 'atr': [1.0] * 9}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, ResonanceParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, -1, 0, 0]
    timed = GeneratedStrategy.generate_signals(data.assign(close=100.0), ind.assign(candidate=1), None, ResonanceParams(hold_bars=2)).data
    assert timed['signal'].tolist()[:5] == [0, 0, 1, 1, 0]
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    flat_ind = GeneratedStrategy.indicators(flat, ResonanceParams())
    assert not flat_ind['candidate'].any()
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
