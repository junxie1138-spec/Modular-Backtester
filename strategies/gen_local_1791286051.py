from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuarterlyDiscountParams:
    ma_window: int = 20
    entry_z: float = 1.2
    holding_bars: int = 8

    def __post_init__(self):
        if self.ma_window < 2 or self.holding_bars < 1:
            raise ValueError('Lookback must be at least two and holding period positive.')
        if not np.isfinite(self.entry_z) or self.entry_z <= 0:
            raise ValueError('entry_z must be finite and positive.')


class GeneratedStrategy(BaseStrategy[QuarterlyDiscountParams]):
    strategy_id = 'gen_local_1791286051'

    @classmethod
    def params_type(cls):
        return QuarterlyDiscountParams

    def warmup_bars(self, params: QuarterlyDiscountParams) -> int:
        return int(max(200, params.ma_window))

    def indicators(self, data: pd.DataFrame, params: QuarterlyDiscountParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['std'].where(out['std'] > 0)).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['quarter_window'] = data.index.month.isin([3, 6, 9, 12]) & (data.index.day >= 25)
        out['bull_regime'] = (close > out['ma200']).fillna(False)
        out['entry'] = (out['quarter_window'] & out['bull_regime'] & (out['z'] <= -params.entry_z)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: QuarterlyDiscountParams) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        eligible = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if i < exit_bar:
                signal[i] = 1
            elif i == exit_bar:
                # Leave one flat bar so the fixed exit cannot be masked by reentry.
                continue
            elif eligible[i]:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = QuarterlyDiscountParams()
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'entry': np.arange(24) == 0}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, params)
    assert result.data['signal'].tolist() == [0] + [1] * 8 + [0] * 15
    assert (result.data['size'] > 0).all()
