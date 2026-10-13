from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonalityParams:
    path_bars: int = 5
    area_threshold: float = 0.15
    hold_bars: int = 2

    def __post_init__(self):
        if self.path_bars < 2:
            raise ValueError('path_bars must be at least two')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two')
        if not np.isfinite(self.area_threshold) or not 0.0 <= self.area_threshold <= 1.0:
            raise ValueError('area_threshold must be between zero and one')


class GeneratedStrategy(BaseStrategy[SignatureSeasonalityParams]):
    strategy_id = 'gen_local_1791197031'

    @classmethod
    def params_type(cls):
        return SignatureSeasonalityParams

    def warmup_bars(self, params: SignatureSeasonalityParams) -> int:
        return int(max(200, params.path_bars + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonalityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['log_return'] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        length = params.path_bars
        out['path_return'] = out['log_return'].rolling(length, min_periods=length).sum()
        out['absolute_return_sum'] = out['log_return'].abs().rolling(length, min_periods=length).sum()
        # Antisymmetric level-two signature: elapsed time versus cumulative return.
        area = pd.Series(0.0, index=data.index)
        for lag in range(length):
            area = area + 0.5 * (length - 1 - 2 * lag) * out['log_return'].shift(lag)
        out['signature_area'] = area
        denominator = 0.5 * (length - 1) * out['absolute_return_sum']
        out['normalized_area'] = (area / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        out['ma_200'] = close.rolling(200, min_periods=200).mean()
        out['monday'] = data.index.dayofweek == 0
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureSeasonalityParams) -> SignalFrame:
        entry = (
            indicators['monday']
            & (indicators['normalized_area'] < -params.area_threshold)
            & (indicators['path_return'] < 0.0)
            & (data['close'] > indicators['ma_200'])
        ).fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    raw[i] = 1
                else:
                    # Force a flat target; ignore entry conditions on this bar.
                    exit_at = -1
            elif entry[i]:
                raw[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=8, freq='D')
    data = pd.DataFrame({'close': 101.0}, index=index)
    indicators = pd.DataFrame({
        'monday': True,
        'normalized_area': -0.5,
        'path_return': -0.01,
        'ma_200': 100.0,
    }, index=index)
    for hold, expected in (
        (1, [0, 1, 0, 1, 0, 1, 0, 1]),
        (2, [0, 1, 1, 0, 1, 1, 0, 1]),
    ):
        result = strategy.generate_signals(data, indicators, None, SignatureSeasonalityParams(hold_bars=hold))
        assert result.data['signal'].tolist() == expected
        assert (result.data['size'] > 0).all()
    falling = pd.DataFrame({'close': np.exp(-0.01 * np.arange(8) ** 2)}, index=index)
    calculated = strategy.indicators(falling, SignatureSeasonalityParams())
    assert calculated['normalized_area'].iloc[-1] < 0
    assert calculated['normalized_area'].iloc[:5].isna().all()


if __name__ == '__main__':
    _self_check()
