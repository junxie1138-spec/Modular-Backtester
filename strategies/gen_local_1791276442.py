from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    breakout_bars: int = 20
    work_bars: int = 5
    percentile_bars: int = 120
    percentile: float = 0.8
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.breakout_bars, self.work_bars, self.percentile_bars, self.holding_bars) < 1:
            raise ValueError('Lookbacks and holding_bars must be positive.')
        if not 0.0 < self.percentile < 1.0:
            raise ValueError('percentile must lie strictly between zero and one.')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791276442'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.breakout_bars + 1, params.work_bars + params.percentile_bars, params.percentile_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['ceiling'] = data['high'].shift(1).rolling(params.breakout_bars, min_periods=params.breakout_bars).max()
        volume = data['volume'].where(data['volume'] > 0)
        out['signed_work'] = (data['close'] - data['open']) * volume
        out['range_work'] = (data['high'] - data['low']).clip(lower=0) * volume
        out['net_work'] = out['signed_work'].rolling(params.work_bars, min_periods=params.work_bars).sum()
        out['total_work'] = out['range_work'].rolling(params.work_bars, min_periods=params.work_bars).sum()
        out['retention'] = out['net_work'] / out['total_work'].where(out['total_work'] > 0)
        out['retention_threshold'] = out['retention'].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.percentile)
        out['volume_threshold'] = volume.shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.percentile)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[['ceiling', 'retention', 'retention_threshold', 'volume_threshold']].notna().all(axis=1)
        out['entry'] = (
            ready
            & (data['close'] > out['ceiling'])
            & (data['close'] > data['open'])
            & (data['volume'] > out['volume_threshold'])
            & (out['retention'] > 0)
            & (out['retention'] > out['retention_threshold'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for i in range(len(data)):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force an exit even when another entry qualifies on this bar.
                    exit_bar = -1
            elif entries[i]:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = Params()
    index = pd.date_range('2020-01-01', periods=3 * params.holding_bars + 5, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'entry': True}, index=index)
    result = GeneratedStrategy().generate_signals(data, entries, None, params).data
    expected = np.zeros(len(data), dtype=int)
    for start in range(1, len(data), params.holding_bars + 1):
        expected[start:start + params.holding_bars] = 1
    assert np.array_equal(result['signal'].to_numpy(), expected)
    assert (result['size'] > 0).all()
