from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DiscountParams:
    ma_window: int = 15
    smoothing: int = 3
    history_window: int = 80
    entry_percentile: float = 0.35

    def __post_init__(self):
        if self.ma_window < 2 or self.smoothing < 1 or self.history_window < 20:
            raise ValueError('Invalid lookback lengths')
        if not 0.0 < self.entry_percentile < 0.5:
            raise ValueError('entry_percentile must be between zero and 0.5')


class GeneratedStrategy(BaseStrategy[DiscountParams]):
    strategy_id = 'gen_local_1791329593'

    @classmethod
    def params_type(cls) -> type[DiscountParams]:
        return DiscountParams

    def warmup_bars(self, params: DiscountParams) -> int:
        return int(params.ma_window + params.smoothing + params.history_window + 1)

    def indicators(self, data: pd.DataFrame, params: DiscountParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['scale'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['raw_z'] = ((close - out['ma']) / out['scale'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['z'] = out['raw_z'].rolling(params.smoothing, min_periods=params.smoothing).mean()
        out['ma_slope'] = out['ma'].diff()
        minimum = max(5, params.history_window // 5)
        out['rising_cutoff'] = out['z'].where(out['ma_slope'] > 0.0).shift(1).rolling(params.history_window, min_periods=minimum).quantile(params.entry_percentile)
        out['falling_cutoff'] = out['z'].where(out['ma_slope'] <= 0.0).shift(1).rolling(params.history_window, min_periods=minimum).quantile(params.entry_percentile)
        out['cutoff'] = out['rising_cutoff'].where(out['ma_slope'] > 0.0, out['falling_cutoff']).where(out['ma_slope'].notna())
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DiscountParams) -> SignalFrame:
        valid = np.isfinite(indicators[['z', 'cutoff']]).all(axis=1)
        # Exit only when a valid observation flips the entry predicate.
        predicate = (indicators['z'] <= indicators['cutoff']).astype(float)
        position = predicate.where(valid).ffill().fillna(0.0)
        df = pd.DataFrame(index=data.index)
        df['signal'] = position.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'z': [-2.0, np.nan, -1.0, 1.0, -2.0, 1.0], 'cutoff': [0.0] * 6}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, DiscountParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 1]
    assert (result['size'] > 0.0).all()


if __name__ == '__main__':
    _self_check()
