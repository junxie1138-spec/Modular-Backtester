from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    min_drawdown: float = 0.01
    min_contraction: float = 0.5
    hold_bars: int = 2
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1791279504'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window, 8) + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if params.peak_window < 2 or not 0 < params.min_drawdown < 1:
            raise ValueError('Invalid drawdown parameters')
        if not 0 < params.min_contraction < 1:
            raise ValueError('min_contraction must be between zero and one')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['log_close'] = np.log(close)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = close / out['peak'] - 1.0
        # Causal Haar details: recent half-block minus preceding half-block.
        for scale in (2, 4, 8):
            half = scale // 2
            out[f'mean_{half}'] = out['log_close'].rolling(half, min_periods=half).mean()
            out[f'haar_{scale}'] = (out[f'mean_{half}'] - out[f'mean_{half}'].shift(half)) * np.sqrt(scale) / 2.0
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2)
        out['previous_fine'] = out['haar_2'].shift(1)
        out['previous_fine_energy'] = out['energy_2'].shift(1)
        denominator = out['previous_fine_energy'].where(out['previous_fine_energy'] > 0)
        out['energy_ratio'] = out['energy_2'] / denominator
        out['contraction'] = 1.0 - out['energy_ratio']
        out['repair_ratio'] = out['haar_2'] / (-out['previous_fine']).where(out['previous_fine'] < 0)
        out['strength'] = (0.5 * out['contraction'].clip(0, 1) + 0.5 * out['repair_ratio'].clip(0, 1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        required = ['drawdown', 'haar_2', 'haar_4', 'haar_8', 'previous_fine', 'contraction', 'repair_ratio']
        ready = np.isfinite(out[required]).all(axis=1)
        out['entry'] = (ready & (out['drawdown'] <= -params.min_drawdown) & (out['previous_fine'] < 0) & (out['haar_2'] > 0) & (out['haar_4'] < 0) & (out['haar_8'] < 0) & (out['contraction'] >= params.min_contraction)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two')
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1')
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strengths = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        remaining = 0
        entry_size = params.min_size
        for i in range(len(data)):
            if remaining > 0:
                remaining -= 1
                if remaining > 0:
                    signals[i] = 1
                    sizes[i] = entry_size
                # Always spend the scheduled exit bar flat, even if entry is true.
            elif entries[i]:
                remaining = params.hold_bars
                entry_size = params.min_size + (params.max_size - params.min_size) * strengths[i]
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [True, True, True, True, True, True, False], 'strength': [0.2, 0.9, 0.9, 0.8, 0.1, 0.1, 0.0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, RecoveryParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert np.allclose(result['size'].iloc[[1, 2, 4, 5]], [0.39, 0.39, 0.81, 0.81])
    result = strategy.generate_signals(data, indicators, None, RecoveryParams(hold_bars=1)).data
    assert result['signal'].tolist() == [0, 1, 0, 1, 0, 1, 0]
    flat = strategy.indicators(data, RecoveryParams(peak_window=2))
    assert not flat['entry'].any()
    assert np.isfinite(flat['strength']).all()
