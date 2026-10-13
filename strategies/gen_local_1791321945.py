from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    volume_window: int = 15
    min_drawdown: float = 0.02
    recovery_volume: float = 1.25
    impact_ratio: float = 0.55
    min_rebound: float = 0.002
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791321945'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        if params.peak_window < 2 or params.volume_window < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        thresholds = (params.min_drawdown, params.recovery_volume, params.impact_ratio, params.min_rebound)
        if not all(np.isfinite(value) for value in thresholds):
            raise ValueError('Thresholds must be finite.')
        if not 0 < params.min_drawdown < 1 or params.recovery_volume < 1:
            raise ValueError('Invalid drawdown or volume threshold.')
        if not 0 < params.impact_ratio < 1 or params.min_rebound <= 0:
            raise ValueError('Invalid impact or rebound threshold.')
        return int(max(params.peak_window + 1, params.volume_window + 3))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        out['volume_baseline'] = volume.rolling(params.volume_window, min_periods=params.volume_window).mean().shift(1)
        out['relative_volume'] = volume / out['volume_baseline'].where(out['volume_baseline'] > 0)
        out['selling_impact'] = (-out['return']).clip(lower=0) / out['relative_volume'].where(out['relative_volume'] > 0)
        out['prior_peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max().shift(1)
        out['prior_drawdown'] = 1.0 - close.shift(1) / out['prior_peak']
        out['previous_return'] = out['return'].shift(1)
        out['earlier_return'] = out['return'].shift(2)
        out['previous_impact'] = out['selling_impact'].shift(1)
        out['earlier_impact'] = out['selling_impact'].shift(2)
        out['previous_volume'] = volume.shift(1)
        out['earlier_volume'] = volume.shift(2)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['volume_recovery'] = ((out['return'] >= params.min_rebound) & (out['relative_volume'] >= params.recovery_volume)).fillna(False)
        out['supply_exhaustion'] = ((out['previous_return'] < 0) & (out['earlier_return'] < 0) & (out['previous_volume'] >= out['earlier_volume']) & (out['previous_impact'] <= params.impact_ratio * out['earlier_impact'])).fillna(False)
        out['in_drawdown'] = (out['prior_drawdown'] >= params.min_drawdown).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        warmup = GeneratedStrategy.warmup_bars(params)
        entries = (indicators['volume_recovery'].fillna(False) & indicators['supply_exhaustion'].fillna(False) & indicators['in_drawdown'].fillna(False)).to_numpy(dtype=bool, copy=True)
        entries[:warmup - 1] = False
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, entry in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif entry:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RecoveryParams(peak_window=2, volume_window=2)
    index = pd.date_range('2020-01-01', periods=11, freq='D')
    data = pd.DataFrame(index=index)
    flags = pd.DataFrame({'volume_recovery': True, 'supply_exhaustion': True, 'in_drawdown': True}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, flags, None, params).data
    assert result['signal'].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    assert result['size'].gt(0).all()
