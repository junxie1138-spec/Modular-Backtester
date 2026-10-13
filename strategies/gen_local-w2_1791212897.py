from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnchorParams:
    ma_window: int = 4
    scale_window: int = 20
    compression_window: int = 3
    compression_limit: float = 0.75
    z_threshold: float = 0.45
    deformation_limit: float = 0.35


class GeneratedStrategy(BaseStrategy[AnchorParams]):
    strategy_id = 'gen_local-w2_1791212897'

    @classmethod
    def params_type(cls) -> type[AnchorParams]:
        return AnchorParams

    def warmup_bars(self, params: AnchorParams) -> int:
        return int(max(params.scale_window + 1,
                       params.ma_window + params.compression_window,
                       params.compression_window) + 1)

    def indicators(self, data: pd.DataFrame, params: AnchorParams) -> pd.DataFrame:
        if min(params.ma_window, params.scale_window, params.compression_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not all(np.isfinite(v) and v > 0 for v in
                   (params.compression_limit, params.z_threshold, params.deformation_limit)):
            raise ValueError('Thresholds must be finite and positive.')
        out = pd.DataFrame(index=data.index)
        out['anchor'] = data['close'].rolling(params.ma_window).mean()
        out['scale'] = data['close'].rolling(params.scale_window).std(ddof=0).shift(1)
        scale = out['scale'].where(out['scale'] > 0)
        out['z'] = (data['close'] - out['anchor']) / scale
        out['bar_range'] = (data['high'] - data['low']).clip(lower=0)
        out['short_range'] = out['bar_range'].rolling(params.compression_window).mean()
        out['reference_range'] = out['bar_range'].rolling(params.scale_window).mean().shift(1)
        reference = out['reference_range'].where(out['reference_range'] > 0)
        out['compression'] = out['short_range'] / reference
        out['anchor_change'] = out['anchor'].diff(params.compression_window)
        out['deformation'] = out['anchor_change'].abs() / scale
        out = out.replace([np.inf, -np.inf], np.nan)
        out['eligible'] = (
            out[['z', 'compression', 'deformation']].notna().all(axis=1)
            & out['compression'].le(params.compression_limit)
            & out['deformation'].le(params.deformation_limit)
        )
        out['entry_direction'] = np.select(
            [out['eligible'] & out['z'].le(-params.z_threshold),
             out['eligible'] & out['z'].ge(params.z_threshold)],
            [1, -1], default=0
        ).astype(int)
        direction = out['entry_direction']
        out['confirmed_direction'] = direction.where(
            direction.eq(direction.shift(1)) & direction.ne(0), 0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnchorParams) -> SignalFrame:
        confirmed = indicators['confirmed_direction'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(confirmed):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Exit on the opposite confirmed entry; leave this decision bar flat.
                position = 0
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = AnchorParams()
    index = pd.date_range('2020-01-01', periods=60)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived['confirmed_direction'].eq(0).all()
    assert strategy.warmup_bars(params) >= params.scale_window + 2
    sample = data.iloc[:9]
    decisions = pd.DataFrame({'confirmed_direction': [0, 1, 0, 0, -1, -1, 0, 0, 1]},
                             index=sample.index)
    result = strategy.generate_signals(sample, decisions, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, -1]
    assert result['size'].gt(0).all()
