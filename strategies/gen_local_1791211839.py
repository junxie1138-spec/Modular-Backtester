from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureBreakoutParams:
    signature_bars: int = 8
    breakout_bars: int = 20
    volume_bars: int = 20
    area_threshold: float = 0.10
    holding_bars: int = 4

    def __post_init__(self):
        if min(self.signature_bars, self.breakout_bars, self.volume_bars) < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 3 <= self.holding_bars <= 5:
            raise ValueError('Holding period must be three to five bars.')
        if not 0.0 <= self.area_threshold < 0.5:
            raise ValueError('Area threshold must be in [0, 0.5).')


class GeneratedStrategy(BaseStrategy[SignatureBreakoutParams]):
    strategy_id = 'gen_local_1791211839'

    @classmethod
    def params_type(cls):
        return SignatureBreakoutParams

    def warmup_bars(self, params: SignatureBreakoutParams) -> int:
        return int(max(200, params.breakout_bars + 1,
                       params.signature_bars + 1,
                       params.volume_bars + params.signature_bars))

    def indicators(self, data: pd.DataFrame, params: SignatureBreakoutParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['log_return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out['volume_baseline'] = volume.rolling(params.volume_bars, min_periods=params.volume_bars).mean().shift(1)
        out['relative_volume'] = (volume / out['volume_baseline'].where(out['volume_baseline'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['prior_high'] = data['high'].rolling(params.breakout_bars, min_periods=params.breakout_bars).max().shift(1)

        x = pd.Series(0.0, index=data.index)
        y = x.copy()
        area = x.copy()
        # Antisymmetric level-two signature; positive means volume leads returns.
        for lag in range(params.signature_bars - 1, -1, -1):
            dx = out['log_return'].shift(lag)
            dy = out['relative_volume'].shift(lag)
            area = area + 0.5 * (y * dx - x * dy)
            x = x + dx
            y = y + dy
        out['path_return'] = x
        out['path_volume'] = y
        out['absolute_return_sum'] = out['log_return'].abs().rolling(params.signature_bars, min_periods=params.signature_bars).sum()
        out['signature_area'] = area
        denominator = out['absolute_return_sum'] * out['path_volume']
        out['normalized_area'] = (area / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureBreakoutParams) -> SignalFrame:
        ready = (
            (data['close'] > indicators['ma200'])
            & (data['close'] > indicators['prior_high'])
            & (indicators['normalized_area'] > params.area_threshold)
            & (indicators['path_return'] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, candidate in enumerate(ready):
            if exit_bar >= 0:
                if i < exit_bar:
                    signals[i] = 1
                else:
                    # Force a flat bar before considering another entry.
                    exit_bar = -1
            elif candidate:
                signals[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = SignatureBreakoutParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'ma200': 90.0, 'prior_high': 99.0,
                               'normalized_area': 0.2, 'path_return': 0.1}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert result['size'].gt(0).all()
    indicators['normalized_area'] = np.nan
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].eq(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['normalized_area'].isna().all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
