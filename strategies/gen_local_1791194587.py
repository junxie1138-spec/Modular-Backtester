from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    baseline_window: int = 24
    energy_jump: float = 1.5

    def __post_init__(self):
        if self.baseline_window < 2:
            raise ValueError('baseline_window must be at least 2')
        if not np.isfinite(self.energy_jump) or self.energy_jump <= 1.0:
            raise ValueError('energy_jump must be finite and greater than 1')


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1791194587'

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(params.baseline_window + 12)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(np.isfinite(close) & (close > 0)))
        out['return'] = out['log_close'].diff()
        out['return_sum_2'] = out['return'].rolling(2, min_periods=2).sum()
        out['return_sum_4'] = out['return'].rolling(4, min_periods=4).sum()
        out['haar_2'] = (out['return'] - out['return'].shift(1)) / np.sqrt(2.0)
        out['haar_4'] = (out['return_sum_2'] - out['return_sum_2'].shift(2)) / 2.0
        out['haar_8'] = (out['return_sum_4'] - out['return_sum_4'].shift(4)) / np.sqrt(8.0)
        for scale in (2, 4, 8):
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2).rolling(4, min_periods=4).mean()
        out['fine_energy'] = (out['energy_2'] + out['energy_4']) / 2.0
        out['coarse_baseline'] = out['energy_8'].rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean().shift(1)
        out['energy_ratio'] = out['energy_8'].div(
            out['coarse_baseline'].where(out['coarse_baseline'] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out['changepoint'] = (
            (out['energy_ratio'] >= params.energy_jump)
            & (out['energy_8'] > out['fine_energy'])
            & out['return_sum_4'].notna()
        ).fillna(False)
        out['entry'] = np.where(
            out['changepoint'], np.sign(out['return_sum_4'].fillna(0.0)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # Opposing qualified entry exits; subsequent confirmation can re-enter.
                position = 0
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = WaveletParams()
    index = pd.date_range('2020-01-01', periods=80, freq='D')
    flat = pd.DataFrame(
        {'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1000.0},
        index=index,
    )
    derived = GeneratedStrategy.indicators(None, flat, params)
    assert derived['entry'].eq(0).all()
    sample = flat.iloc[:7]
    entries = pd.DataFrame({'entry': [1, 0, -1, -1, 0, 1, 1]}, index=sample.index)
    result = GeneratedStrategy.generate_signals(None, sample, entries, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result.data['size'].gt(0).all()
