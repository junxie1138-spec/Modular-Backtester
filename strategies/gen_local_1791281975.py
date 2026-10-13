from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRelaxationParams:
    ma_window: int = 20
    percentile_window: int = 126
    entry_percentile: float = 0.85
    max_retained_fraction: float = 0.65
    profit_target: float = 0.012
    hold_bars: int = 4

    def __post_init__(self):
        if self.ma_window < 2 or self.percentile_window < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0.5 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between 0.5 and 1.')
        if not 0.0 < self.max_retained_fraction < 1.0:
            raise ValueError('max_retained_fraction must be between 0 and 1.')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be positive and finite.')
        if self.hold_bars not in (3, 4, 5):
            raise ValueError('hold_bars must be three, four, or five.')


class GeneratedStrategy(BaseStrategy[GapRelaxationParams]):
    strategy_id = 'gen_local_1791281975'

    @classmethod
    def params_type(cls):
        return GapRelaxationParams

    def warmup_bars(self, params: GapRelaxationParams) -> int:
        return int(params.ma_window + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapRelaxationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean().shift(1)
        out['std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0).shift(1).replace(0.0, np.nan)
        out['gap'] = data['open'] - out['previous_close']
        out['open_z'] = (data['open'] - out['ma']) / out['std']
        out['close_z'] = (close - out['ma']) / out['std']
        out['abs_open_z'] = out['open_z'].abs()
        out['abs_close_z'] = out['close_z'].abs()
        out['opening_threshold'] = out['abs_open_z'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)
        out['residual_threshold'] = out['abs_close_z'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(0.5)
        out['retained_fraction'] = out['abs_close_z'] / out['abs_open_z'].replace(0.0, np.nan)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & (data['open'] > 0) & (close > 0)
        eligible = (
            ready
            & (out['abs_open_z'] > out['opening_threshold'])
            & (out['gap'] * out['open_z'] > 0)
            & (out['open_z'] * out['close_z'] > 0)
            & (out['retained_fraction'] <= params.max_retained_fraction)
            & (out['abs_close_z'] > out['residual_threshold'])
        )
        out['entry'] = np.where(eligible, -np.sign(out['open_z'].fillna(0.0)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRelaxationParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # A decision from the preceding close fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    # Targets are observed at the close; exits fill next open.
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        position = 0
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=14)
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[2], 'close'] = 102.0
    data.loc[index[7], 'close'] = 98.0
    indicators = pd.DataFrame({'entry': np.zeros(14, dtype=int)}, index=index)
    indicators.loc[index[[0, 4, 8]], 'entry'] = [1, -1, 1]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, GapRelaxationParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, -1, 0, 1, 1, 1, 1, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
