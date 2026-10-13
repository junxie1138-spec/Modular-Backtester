from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    innovation_threshold: float = 0.08
    profit_target: float = 0.015

    def __post_init__(self):
        if not np.isfinite(self.innovation_threshold) or self.innovation_threshold <= 0:
            raise ValueError('innovation_threshold must be finite and positive')
        if not np.isfinite(self.profit_target) or not 0 < self.profit_target < 1:
            raise ValueError('profit_target must be between zero and one')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791314938'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 14

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['midpoint'] = (data['high'] + data['low']) / 2.0
        t = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        for window, name in ((4, 'fast_slope'), (12, 'slow_slope')):
            mean_price = out['midpoint'].rolling(window, min_periods=window).mean()
            mean_time = t.rolling(window, min_periods=window).mean()
            mean_product = (out['midpoint'] * t).rolling(window, min_periods=window).mean()
            out[name] = (mean_product - mean_price * mean_time) / ((window * window - 1) / 12.0)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(12, min_periods=12).mean()
        out['predicted_slope'] = out['slow_slope'].shift(1)
        out['innovation'] = (out['fast_slope'] - out['predicted_slope']) / out['atr'].shift(1).replace(0.0, np.nan)
        out['upper_boundary'] = data['high'].rolling(12, min_periods=12).max().shift(1) + out['predicted_slope']
        out['lower_boundary'] = data['low'].rolling(12, min_periods=12).min().shift(1) + out['predicted_slope']
        valid = np.isfinite(out[['innovation', 'predicted_slope', 'upper_boundary', 'lower_boundary']]).all(axis=1)
        long_entry = valid & (out['predicted_slope'] > 0) & (out['innovation'] > params.innovation_threshold) & (data['close'] > out['upper_boundary'])
        short_entry = valid & (out['predicted_slope'] < 0) & (out['innovation'] < -params.innovation_threshold) & (data['close'] < out['lower_boundary'])
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        candidates = indicators['entry'].fillna(0).to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (close[i] / entry_price - 1.0) if valid_entry else np.nan
                # Evaluate targets at the close; the exit executes next bar.
                if not valid_entry or gain >= params.profit_target or i - entry_bar + 1 >= 4:
                    position = 0
            elif candidates[i] and np.isfinite(close[i]) and close[i] > 0:
                position = int(candidates[i])
                entry_bar = i + 1
            raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(profit_target=0.01)
    index = pd.date_range('2020-01-01', periods=8)
    for direction, second_close, expected in (
        (1, 100.0, [0, 1, 1, 1, 1, 0, 0, 0]),
        (1, 102.0, [0, 1, 0, 0, 0, 0, 0, 0]),
        (-1, 98.0, [0, -1, 0, 0, 0, 0, 0, 0]),
    ):
        data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 97.0, 'close': 100.0, 'volume': 1000.0}, index=index)
        data.loc[index[1], 'close'] = second_close
        indicators = pd.DataFrame({'entry': [direction, 0, 0, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params)
        assert result.data['signal'].tolist() == expected
        assert (result.data['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
