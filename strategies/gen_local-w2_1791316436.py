from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEnergyParams:
    energy_window: int = 6
    baseline_window: int = 60
    energy_ceiling: float = 0.55
    range_ceiling: float = 0.85
    close_floor: float = 0.65
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangeEnergyParams]):
    strategy_id = 'gen_local-w2_1791316436'

    @classmethod
    def params_type(cls) -> type[RangeEnergyParams]:
        return RangeEnergyParams

    @staticmethod
    def warmup_bars(params: RangeEnergyParams) -> int:
        return int(max(40, 16 + params.energy_window + params.baseline_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeEnergyParams) -> pd.DataFrame:
        if params.energy_window < 1 or params.baseline_window < 1 or params.hold_bars < 1:
            raise ValueError('Window lengths and hold_bars must be positive.')
        if not (0 < params.energy_ceiling <= 1 and 0 < params.range_ceiling <= 1):
            raise ValueError('Compression thresholds must be in (0, 1].')
        if not 0 <= params.close_floor <= 1:
            raise ValueError('close_floor must be in [0, 1].')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        width = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['relative_range'] = width / close
        ratios = []
        for scale in (2, 4, 8):
            recent = out['relative_range'].rolling(scale, min_periods=scale).mean()
            out[f'haar_{scale}'] = np.sqrt(scale / 2.0) * (recent - recent.shift(scale))
            out[f'energy_{scale}'] = out[f'haar_{scale}'].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            out[f'baseline_{scale}'] = out[f'energy_{scale}'].rolling(
                params.baseline_window, min_periods=params.baseline_window
            ).mean().shift(1)
            name = f'energy_ratio_{scale}'
            out[name] = out[f'energy_{scale}'] / out[f'baseline_{scale}'].clip(lower=1e-12)
            ratios.append(name)
        out['compression_score'] = out[ratios].max(axis=1, skipna=False)
        out['range_fast'] = out['relative_range'].rolling(4, min_periods=4).mean()
        out['range_slow'] = out['relative_range'].rolling(40, min_periods=40).mean()
        out['range_ratio'] = out['range_fast'] / out['range_slow'].where(out['range_slow'] > 0)
        out['close_position'] = (close - data['low']) / width
        out['advance'] = close.diff()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeEnergyParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError('hold_bars must be positive.')
        eligible = (
            (indicators['compression_score'] < params.energy_ceiling)
            & (indicators['range_ratio'] < params.range_ceiling)
            & (indicators['close_position'] >= params.close_floor)
            & (indicators['advance'] > 0)
        ).fillna(False)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                else:
                    signals[i] = 1
            elif confirmed[i]:
                entry_bar = i
                signals[i] = 1
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'compression_score': 0.1,
        'range_ratio': 0.5,
        'close_position': 0.9,
        'advance': 1.0,
    }, index=index)
    result = GeneratedStrategy.generate_signals(
        data, indicators, None, RangeEnergyParams(hold_bars=3)
    )
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 1, 0, 1, 1, 1, 0]
