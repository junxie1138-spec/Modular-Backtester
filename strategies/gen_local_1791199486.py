from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    breakout_window: int = 20
    energy_window: int = 40
    coarse_share: float = 0.65
    energy_multiple: float = 1.25
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1791199486'

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(params.breakout_window + 2, 16 + params.energy_window + 2))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.breakout_window < 2 or params.energy_window < 2 or params.hold_bars < 1:
            raise ValueError('Lookbacks must be at least two and hold_bars must be positive.')
        if not 0.0 < params.coarse_share < 1.0 or params.energy_multiple <= 0.0:
            raise ValueError('Invalid energy thresholds.')
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        # Causal Haar details: adjacent, non-overlapping blocks ending at this bar.
        for half_width in (2, 4, 8):
            mean = out['log_close'].rolling(half_width, min_periods=half_width).mean()
            out[f'mean_{half_width}'] = mean
            out[f'detail_{half_width}'] = np.sqrt(half_width / 2.0) * (mean - mean.shift(half_width))
            out[f'energy_{half_width}'] = out[f'detail_{half_width}'].pow(2)
        out['coarse_energy'] = out['energy_4'] + out['energy_8']
        out['total_energy'] = out['energy_2'] + out['coarse_energy']
        out['coarse_share'] = out['coarse_energy'].div(out['total_energy'].where(out['total_energy'] > 0))
        out['energy_baseline'] = out['coarse_energy'].rolling(
            params.energy_window, min_periods=params.energy_window
        ).median().shift(1)
        out['upper_channel'] = data['high'].rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).max().shift(1)
        out['lower_channel'] = data['low'].rolling(
            params.breakout_window, min_periods=params.breakout_window
        ).min().shift(1)
        out['energy_ready'] = (
            out['coarse_share'].ge(params.coarse_share)
            & out['energy_baseline'].gt(0)
            & out['coarse_energy'].gt(params.energy_multiple * out['energy_baseline'])
        )
        bullish = (
            out['energy_ready']
            & out['detail_4'].gt(0)
            & out['detail_8'].gt(0)
            & data['close'].gt(out['upper_channel'])
        )
        bearish = (
            out['energy_ready']
            & out['detail_4'].lt(0)
            & out['detail_8'].lt(0)
            & data['close'].lt(out['lower_channel'])
        )
        out['candidate'] = np.select([bullish, bearish], [1, -1], default=0).astype(int)
        out['confirmed'] = out['candidate'].where(
            out['candidate'].eq(out['candidate'].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError('hold_bars must be positive.')
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        candidates = indicators['confirmed'].fillna(0).to_numpy(dtype=int)
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i - entry_bar >= params.hold_bars:
                    position = 0
                    # Leave an explicit flat bar; no immediate re-entry at expiry.
                signals[i] = position
                continue
            if candidate in (-1, 1):
                position = int(candidate)
                entry_bar = i
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=44, freq='B')
    data = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = WaveletParams()
    flat_indicators = strategy.indicators(data, params)
    assert flat_indicators['confirmed'].eq(0).all()
    scripted = pd.DataFrame({'confirmed': 0}, index=index)
    scripted.iloc[1, 0] = 1
    scripted.iloc[5, 0] = -1
    scripted.iloc[22, 0] = -1
    signals = strategy.generate_signals(data, scripted, None, params).data
    expected = np.zeros(44, dtype=int)
    expected[2:20] = 1
    expected[23:41] = -1
    assert np.array_equal(signals['signal'].to_numpy(), expected)
    assert signals['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
