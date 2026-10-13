from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    wave_ratio: float = 1.3
    hold_bars: int = 8

    def __post_init__(self):
        if not np.isfinite(self.wave_ratio) or self.wave_ratio <= 1.0:
            raise ValueError('wave_ratio must be finite and greater than one')
        if isinstance(self.hold_bars, bool) or not isinstance(self.hold_bars, int) or self.hold_bars < 1:
            raise ValueError('hold_bars must be a positive integer')


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = 'gen_local-w2_1791249926'

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    @staticmethod
    def warmup_bars(params: WaveParams) -> int:
        return 16

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['vol_recent'] = out['return'].rolling(5, min_periods=5).std(ddof=0)
        out['vol_middle'] = out['vol_recent'].shift(5)
        out['vol_old'] = out['vol_recent'].shift(10)
        out['displacement'] = close.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan)
        valid = np.isfinite(out[['vol_recent', 'vol_middle', 'vol_old', 'displacement']]).all(axis=1)
        valid &= out[['vol_recent', 'vol_middle', 'vol_old']].gt(0).all(axis=1)
        out['fresh_front'] = valid & (out['vol_recent'] > params.wave_ratio * out['vol_middle']) & (out['vol_middle'] <= out['vol_old'])
        out['dissipating_front'] = valid & (out['vol_middle'] > params.wave_ratio * out['vol_old']) & (out['vol_recent'] < out['vol_middle'] / params.wave_ratio)
        out['direction'] = np.sign(out['displacement'].fillna(0)).astype(int)
        out['candidate'] = np.select(
            [out['fresh_front'], out['dissipating_front']],
            [out['direction'], -out['direction']],
            default=0,
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(indicators['candidate'].fillna(0).to_numpy()):
            if position:
                if i == exit_bar:
                    position = 0
                else:
                    signal[i] = position
            elif entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
                signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = WaveParams()
    index = pd.date_range('2020-01-01', periods=2 * params.hold_bars + 4)
    data = pd.DataFrame(index=index)
    candidate = np.zeros(len(index), dtype=int)
    candidate[0] = 1
    candidate[1] = -1
    candidate[params.hold_bars] = -1
    candidate[params.hold_bars + 1] = -1
    ind = pd.DataFrame({'candidate': candidate}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    h = params.hold_bars
    assert result['signal'].iloc[0] == 0
    assert result['signal'].iloc[1:h + 1].eq(1).all()
    assert result['signal'].iloc[h + 1] == 0
    assert result['signal'].iloc[h + 2:2 * h + 2].eq(-1).all()
    assert result['signal'].iloc[2 * h + 2] == 0
    assert result['size'].gt(0).all()
