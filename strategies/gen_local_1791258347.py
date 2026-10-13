from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    area_window: int = 4
    range_window: int = 12
    resistance_delay: int = 4
    area_threshold: float = 0.05

    def __post_init__(self):
        if self.area_window < 2 or self.range_window < 2:
            raise ValueError('Rolling windows must be at least two bars.')
        if self.resistance_delay < 1:
            raise ValueError('Resistance must exclude the current bar.')
        if not 0.0 <= self.area_threshold < 0.5:
            raise ValueError('Area threshold must lie in [0, 0.5).')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791258347'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.area_window + 1,
                       params.range_window + params.resistance_delay) + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out['volume_increment'] = data['volume'].where(data['volume'] >= 0).replace([np.inf, -np.inf], np.nan)
        r = out['return']
        v = out['volume_increment']
        length = params.area_window
        area = pd.Series(0.0, index=data.index)
        # Exact second-order signed area: sum over all ordered increment pairs.
        for lag in range(1, length):
            cross = v.shift(lag) * r - r.shift(lag) * v
            area = area + cross.rolling(length - lag, min_periods=length - lag).sum()
        out['raw_area'] = 0.5 * area
        out['volume_total'] = v.rolling(length, min_periods=length).sum()
        out['return_variation'] = r.abs().rolling(length, min_periods=length).sum()
        denominator = out['volume_total'] * out['return_variation']
        out['signature_area'] = (out['raw_area'] / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)
        out['resistance'] = data['high'].shift(params.resistance_delay).rolling(params.range_window, min_periods=params.range_window).max()
        out['entry_condition'] = (
            np.isfinite(out['signature_area'])
            & np.isfinite(out['resistance'])
            & np.isfinite(data['close'])
            & (out['signature_area'] > params.area_threshold)
            & (data['close'] > out['resistance'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        condition = indicators['entry_condition'].fillna(False).astype(bool)
        df = pd.DataFrame(index=data.index)
        # Two true bars enter; the first false bar exits. No timed exit.
        df['signal'] = (condition & condition.shift(1, fill_value=False)).astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry_condition': [False, True, True, True, False, True, True, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SignatureParams())
    assert result.data['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert (result.data['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
