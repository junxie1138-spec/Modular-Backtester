from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 3
    percentile_bars: int = 126
    tail_percentile: float = 0.85
    profit_target: float = 0.006
    hold_bars: int = 2

    def __post_init__(self):
        if self.path_bars < 2 or self.percentile_bars < 20:
            raise ValueError('Lookbacks are too short')
        if not 0.5 < self.tail_percentile < 1.0:
            raise ValueError('tail_percentile must be between 0.5 and 1')
        if not 0.0 < self.profit_target < 1.0:
            raise ValueError('profit_target must be between 0 and 1')
        if self.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791302277'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(params.path_bars + params.percentile_bars + 1)

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['log_return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        length = params.path_bars
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric level-two signature: (integral t dx - x dt) / 2.
        # Each chronological time increment is 1 / length.
        for lag in range(length):
            area = area + ((length - 1 - 2 * lag) / (2.0 * length)) * out['log_return'].shift(lag)
        out['area'] = area
        out['travel'] = out['log_return'].abs().rolling(length, min_periods=length).sum()
        out['signature'] = (out['area'] / out['travel'].where(out['travel'] > 0)).replace([np.inf, -np.inf], np.nan)
        history = out['signature'].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars)
        out['lower'] = history.quantile(1.0 - params.tail_percentile)
        out['upper'] = history.quantile(params.tail_percentile)
        days = data.index.day
        remaining = data.index.days_in_month - days
        out['month_boundary'] = (days <= 4) | (remaining < 4)
        valid = out[['signature', 'lower', 'upper']].notna().all(axis=1)
        buy = valid & out['month_boundary'] & (out['signature'] < out['lower']) & (out['signature'] < 0)
        sell = valid & out['month_boundary'] & (out['signature'] > out['upper']) & (out['signature'] > 0)
        out['entry'] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            # Yesterday's target takes effect at today's open.
            held = int(decisions[i - 1]) if i else 0
            if held != position:
                if held:
                    entry_price = opens[i]
                    entry_bar = i
                else:
                    entry_price = np.nan
                    entry_bar = -1
                position = held
            if position:
                prices_valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if prices_valid else np.nan
                target_hit = prices_valid and gain >= params.profit_target
                time_hit = i - entry_bar + 1 >= params.hold_bars
                # Close-confirmed exits execute at the following open.
                decisions[i] = 0 if target_hit or time_hit or not prices_valid else position
            elif np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = entries[i]
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    dates = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 99.0, 'close': [100.0, 102.0, 100.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=dates)
    indicators = pd.DataFrame({'entry': [1, 0, -1, 0, 0, 0]}, index=dates)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, SignatureParams())
    assert result.data['signal'].tolist() == [0, 1, 0, -1, -1, 0]
    assert (result.data['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
