from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VarianceRepairParams:
    fast_window: int = 8
    slow_window: int = 60
    downside_threshold: float = 0.40
    target_vol: float = 0.12
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VarianceRepairParams]):
    strategy_id = 'gen_local-w2_1791205115'

    @classmethod
    def params_type(cls) -> type[VarianceRepairParams]:
        return VarianceRepairParams

    @staticmethod
    def warmup_bars(params: VarianceRepairParams) -> int:
        return int(max(params.fast_window, params.slow_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VarianceRepairParams) -> pd.DataFrame:
        if not (2 <= params.fast_window < params.slow_window):
            raise ValueError('Require 2 <= fast_window < slow_window')
        if not (0.0 < params.downside_threshold < 0.55):
            raise ValueError('Require 0 < downside_threshold < 0.55')
        if not (np.isfinite(params.target_vol) and params.target_vol > 0):
            raise ValueError('target_vol must be finite and positive')
        out = pd.DataFrame(index=data.index)
        out['return'] = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['squared_return'] = out['return'].pow(2)
        out['downside_energy'] = out['return'].clip(upper=0).pow(2)
        for name, window in [('fast', params.fast_window), ('slow', params.slow_window)]:
            out[name + '_vol'] = out['return'].rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
            out[name + '_energy'] = out['squared_return'].rolling(window, min_periods=window).mean()
            out[name + '_downside_energy'] = out['downside_energy'].rolling(window, min_periods=window).mean()
            out[name + '_downside_share'] = out[name + '_downside_energy'] / out[name + '_energy'].replace(0.0, np.nan)
        out['repair'] = out['slow_downside_share'] - out['fast_downside_share']
        out['strength'] = (out['repair'] / (1.0 - params.downside_threshold)).clip(0.0, 1.0)
        out['risk_vol'] = out[['fast_vol', 'slow_vol']].max(axis=1, skipna=False)
        out['desired_size'] = (params.target_vol / out['risk_vol'].replace(0.0, np.nan) * out['strength']).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95).fillna(0.05)
        out['qualified'] = (
            (out['slow_downside_share'] > 0.55)
            & (out['fast_downside_share'] < params.downside_threshold)
            & (out['fast_vol'] < out['slow_vol'])
            & (out['risk_vol'] > 0.0)
        ).fillna(False)
        out['entry'] = out['qualified'] & ~out['qualified'].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VarianceRepairParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError('hold_bars must be positive')
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), 0.05, dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        desired = indicators['desired_size'].to_numpy(dtype=float)
        entry_bar = -1
        locked_size = 0.05
        for i in range(len(data)):
            if entry_bar >= 0 and i - entry_bar == params.hold_bars:
                entry_bar = -1
                continue
            if entry_bar < 0 and entries[i] and np.isfinite(desired[i]) and desired[i] > 0.0:
                entry_bar = i
                locked_size = float(np.clip(desired[i], 0.05, 0.95))
            if entry_bar >= 0:
                signal[i] = 1
                size[i] = locked_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    sample = pd.DataFrame(index=pd.date_range('2020-01-01', periods=9))
    derived = pd.DataFrame({'entry': True, 'desired_size': np.linspace(0.2, 0.6, 9)}, index=sample.index)
    result = GeneratedStrategy.generate_signals(sample, derived, None, VarianceRepairParams(hold_bars=3)).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0]
    assert np.allclose(result['size'].iloc[1:4], 0.2)
    assert np.allclose(result['size'].iloc[5:8], 0.4)
    assert (result['size'] > 0).all()
