from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRecoveryParams:
    lookback: int = 20
    gap_sigma: float = 1.8
    hold_bars: int = 4
    size: float = 0.95

    def __post_init__(self):
        if self.lookback < 2 or not np.isfinite(self.gap_sigma) or self.gap_sigma <= 0:
            raise ValueError('lookback must be >= 2 and gap_sigma must be positive and finite')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be 3, 4, or 5')
        if not np.isfinite(self.size) or not 0 < self.size <= 1:
            raise ValueError('size must be finite and in (0, 1]')


class GeneratedStrategy(BaseStrategy[GapRecoveryParams]):
    strategy_id = 'gen_local_1791286767'

    @classmethod
    def params_type(cls):
        return GapRecoveryParams

    def warmup_bars(self, params: GapRecoveryParams) -> int:
        return params.lookback + 4

    def indicators(self, data: pd.DataFrame, params: GapRecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['gap'] = (data['open'] / data['close'].shift(1) - 1).replace([np.inf, -np.inf], np.nan)
        out['gap_rms'] = np.sqrt(out['gap'].pow(2).rolling(params.lookback, min_periods=params.lookback).mean()).shift(1)
        out['shock_gap'] = out['gap'].shift(2)
        out['shock_noise'] = out['gap_rms'].shift(2)
        out['anchor'] = data['close'].shift(3)
        out['shock_open'] = data['open'].shift(2)
        out['gap_debt'] = out['anchor'] - out['shock_open']
        out['repair_fraction'] = ((data['close'] - out['shock_open']) / out['gap_debt'].where(out['gap_debt'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['close_step'] = data['close'].diff()
        out['low_step'] = data['low'].diff()
        out['confirmation'] = (
            out['close_step'].gt(0)
            & out['close_step'].shift(1).gt(0)
            & out['low_step'].gt(0)
            & out['low_step'].shift(1).gt(0)
        )
        out['entry'] = (
            out['shock_noise'].gt(0)
            & out['shock_gap'].lt(-params.gap_sigma * out['shock_noise'])
            & out['repair_fraction'].between(0.15, 0.75)
            & out['confirmation']
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRecoveryParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            # Always exit on schedule, including when another entry qualifies.
            if i == exit_at:
                continue
            if i > exit_at and entries[i]:
                exit_at = i + params.hold_bars
            if i < exit_at:
                signal[i] = 1
        df = pd.DataFrame(index=data.index)
        df['signal'] = signal
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    data = pd.DataFrame(index=pd.date_range('2020-01-01', periods=12))
    indicators = pd.DataFrame({'entry': True}, index=data.index)
    for hold in (3, 4, 5):
        params = GapRecoveryParams(hold_bars=hold)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [int(i > 0 and (i - 1) % (hold + 1) < hold) for i in range(len(data))]
        assert result['size'].gt(0).all()
    indicators['entry'] = False
    assert strategy.generate_signals(data, indicators, None, GapRecoveryParams()).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
