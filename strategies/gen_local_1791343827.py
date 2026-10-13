from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapWaveParams:
    energy_window: int = 5
    baseline_window: int = 20
    fine_share: float = 0.65
    restitution_min: float = 1.1
    hold_bars: int = 8

    def __post_init__(self):
        if self.energy_window < 1 or self.baseline_window < 1:
            raise ValueError('Energy windows must be positive.')
        if not 0.0 < self.fine_share < 1.0:
            raise ValueError('fine_share must be between zero and one.')
        if not np.isfinite(self.restitution_min) or self.restitution_min <= 0:
            raise ValueError('restitution_min must be finite and positive.')
        if self.hold_bars < 1:
            raise ValueError('hold_bars must be positive.')


class GeneratedStrategy(BaseStrategy[GapWaveParams]):
    strategy_id = 'gen_local_1791343827'

    @classmethod
    def params_type(cls):
        return GapWaveParams

    def warmup_bars(self, params: GapWaveParams) -> int:
        return int(8 + params.energy_window + params.baseline_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapWaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        open_price = data['open'].where(data['open'] > 0)
        close_price = data['close'].where(data['close'] > 0)
        out['gap'] = np.log(open_price / close_price.shift(1))
        out['session_return'] = np.log(close_price / open_price)
        for scale in (2, 4, 8):
            half = scale // 2
            recent = out['gap'].rolling(half, min_periods=half).mean()
            out[f'haar_{scale}'] = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        energy_columns = ['energy_2', 'energy_4', 'energy_8']
        out['total_energy'] = out[energy_columns].sum(axis=1, min_count=3)
        out['fine_fraction'] = out['energy_2'] / out['total_energy'].where(out['total_energy'] > 0)
        out['energy_reference'] = out['total_energy'].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out['downside_gap_loss'] = (-out['gap'].clip(upper=0)).rolling(3, min_periods=3).sum()
        out['session_restitution'] = out['session_return'].rolling(3, min_periods=3).sum()
        out['restitution_ratio'] = out['session_restitution'] / out['downside_gap_loss'].where(
            out['downside_gap_loss'] > 0
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapWaveParams,
    ) -> SignalFrame:
        wavelet_agrees = (
            (indicators['fine_fraction'] >= params.fine_share)
            & (indicators['energy_reference'] > 0)
            & (indicators['total_energy'] > 1.25 * indicators['energy_reference'])
        )
        restitution_agrees = (
            (indicators['gap'] < 0)
            & (indicators['session_return'] > 0)
            & (indicators['restitution_ratio'] >= params.restitution_min)
        )
        entries = (wavelet_agrees & restitution_agrees).fillna(False).to_numpy(dtype=bool)
        entries[:self.warmup_bars(params)] = False
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif entries[i]:
                entry_bar = i
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = GapWaveParams()
    strategy = GeneratedStrategy()
    warmup = strategy.warmup_bars(params)
    index = pd.date_range('2020-01-01', periods=warmup + 2 * params.hold_bars + 5, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not np.isinf(derived.to_numpy()).any()
    assert strategy.generate_signals(data, derived, None, params).data['signal'].eq(0).all()
    gates = pd.DataFrame({'fine_fraction': 0.9, 'energy_reference': 1.0, 'total_energy': 2.0, 'gap': -0.01, 'session_return': 0.02, 'restitution_ratio': 2.0}, index=index)
    for missing_gate in ('fine_fraction', 'restitution_ratio'):
        blocked = gates.copy()
        blocked[missing_gate] = 0.0
        assert strategy.generate_signals(data, blocked, None, params).data['signal'].eq(0).all()
    gates.loc[index[warmup + 1]:, 'restitution_ratio'] = 0.0
    result = strategy.generate_signals(data, gates, None, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[warmup + 1:warmup + 1 + params.hold_bars] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert result['size'].gt(0).all()
