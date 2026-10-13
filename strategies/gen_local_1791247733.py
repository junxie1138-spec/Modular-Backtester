from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    history_bars: int = 126
    energy_percentile: float = 0.75
    holding_bars: int = 8

    def __post_init__(self):
        if self.history_bars < 8:
            raise ValueError('history_bars must be at least 8')
        if not 0.0 < self.energy_percentile < 1.0:
            raise ValueError('energy_percentile must be between zero and one')
        if self.holding_bars < 1:
            raise ValueError('holding_bars must be positive')


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = 'gen_local_1791247733'

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(params.history_bars + 9)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['log_close'] = np.log(close.where(close > 0.0)).replace([np.inf, -np.inf], np.nan)
        for width in (1, 2, 4):
            out[f'mean_{width}'] = out['log_close'].rolling(width, min_periods=width).mean()
            out[f'haar_{width}'] = np.sqrt(width / 2.0) * (
                out[f'mean_{width}'] - out[f'mean_{width}'].shift(width)
            )
            out[f'energy_{width}'] = out[f'haar_{width}'].pow(2)
        out['total_energy'] = out[['energy_1', 'energy_2', 'energy_4']].sum(axis=1, min_count=3)
        out['coarse_share'] = out['energy_4'] / out['total_energy'].replace(0.0, np.nan)
        out['share_threshold'] = out['coarse_share'].shift(1).rolling(
            params.history_bars, min_periods=params.history_bars
        ).quantile(params.energy_percentile)
        out['activity_threshold'] = out['total_energy'].shift(1).rolling(
            params.history_bars, min_periods=params.history_bars
        ).quantile(0.5)
        coefficients = out[['haar_1', 'haar_2', 'haar_4']]
        out['up_coherence'] = coefficients.gt(0.0).all(axis=1)
        out['down_coherence'] = coefficients.lt(0.0).all(axis=1)
        out['direction'] = out['up_coherence'].astype(int) - out['down_coherence'].astype(int)
        out['breakout'] = (
            out['coarse_share'].gt(out['share_threshold'])
            & out['coarse_share'].shift(1).le(out['share_threshold'].shift(1))
            & out['total_energy'].gt(out['activity_threshold'])
        )
        out['entry'] = out['direction'].where(out['breakout'], 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position and i == exit_bar:
                position = 0
            elif position == 0 and entry:
                position = int(entry)
                exit_bar = i + params.holding_bars
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=20, freq='B')
    data = pd.DataFrame(100.0, index=index, columns=['open', 'high', 'low', 'close', 'volume'])
    indicators = pd.DataFrame({'entry': np.zeros(20, dtype=int)}, index=index)
    indicators.iloc[0, 0] = 1
    indicators.iloc[2, 0] = -1
    indicators.iloc[9, 0] = -1
    result = GeneratedStrategy().generate_signals(data, indicators, None, WaveletParams())
    assert result.data['signal'].tolist() == [0] + [1] * 8 + [0] + [-1] * 8 + [0, 0]
    assert result.data['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
