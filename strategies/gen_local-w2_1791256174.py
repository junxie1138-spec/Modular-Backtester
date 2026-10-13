from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExcursionParams:
    ma_window: int = 40
    scale_window: int = 30
    memory: int = 6
    stretch: float = 1.5
    recovery: float = 1.0
    min_inventory: float = 1.5
    drift_limit: float = 0.5
    profit_target: float = 0.025
    max_hold: int = 10
    min_size: float = 0.25
    max_size: float = 1.0

    def __post_init__(self):
        if min(self.ma_window, self.scale_window, self.memory) < 2 or self.max_hold < 1:
            raise ValueError('Invalid lookback or holding period')
        values = (self.stretch, self.recovery, self.min_inventory, self.drift_limit,
                  self.profit_target, self.min_size, self.max_size)
        if not all(np.isfinite(v) for v in values):
            raise ValueError('Parameters must be finite')
        if not (0 < self.recovery < self.stretch and self.min_inventory > 0
                and self.drift_limit >= 0 and self.profit_target > 0
                and 0 < self.min_size <= self.max_size <= 1):
            raise ValueError('Invalid thresholds or sizing bounds')


class GeneratedStrategy(BaseStrategy[ExcursionParams]):
    strategy_id = 'gen_local-w2_1791256174'

    @classmethod
    def params_type(cls):
        return ExcursionParams

    def warmup_bars(self, params: ExcursionParams) -> int:
        return int(params.ma_window + params.scale_window + params.memory + 1)

    def indicators(self, data: pd.DataFrame, params: ExcursionParams) -> pd.DataFrame:
        p = params
        out = pd.DataFrame(index=data.index)
        close = data['close'].replace([np.inf, -np.inf], np.nan)
        out['ma'] = close.rolling(p.ma_window, min_periods=p.ma_window).mean()
        out['distance'] = close - out['ma']
        out['scale'] = out['distance'].rolling(p.scale_window, min_periods=p.scale_window).std(ddof=0)
        scale = out['scale'].where(out['scale'] > 0)
        out['z'] = (out['distance'] / scale).replace([np.inf, -np.inf], np.nan)
        out['anchor_drift'] = (out['ma'].diff(p.memory).abs() / scale).replace([np.inf, -np.inf], np.nan)
        out['down_inventory'] = (-out['z'] - p.stretch).clip(lower=0).rolling(p.memory, min_periods=p.memory).sum()
        out['up_inventory'] = (out['z'] - p.stretch).clip(lower=0).rolling(p.memory, min_periods=p.memory).sum()
        out['prior_down_inventory'] = out['down_inventory'].shift(1)
        out['prior_up_inventory'] = out['up_inventory'].shift(1)
        out['prior_z'] = out['z'].shift(1)
        stable = out['anchor_drift'].le(p.drift_limit)
        out['long_entry'] = (stable & out['prior_z'].lt(-p.recovery)
                             & out['z'].ge(-p.recovery) & out['z'].lt(0)
                             & out['prior_down_inventory'].ge(p.min_inventory)).fillna(False)
        out['short_entry'] = (stable & out['prior_z'].gt(p.recovery)
                              & out['z'].le(p.recovery) & out['z'].gt(0)
                              & out['prior_up_inventory'].ge(p.min_inventory)).fillna(False)
        out['long_size'] = p.min_size + (p.max_size - p.min_size) * (out['prior_down_inventory'] / (3 * p.min_inventory)).clip(0, 1)
        out['short_size'] = p.min_size + (p.max_size - p.min_size) * (out['prior_up_inventory'] / (3 * p.min_inventory)).clip(0, 1)
        out[['long_size', 'short_size']] = out[['long_size', 'short_size']].fillna(p.min_size)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ExcursionParams) -> SignalFrame:
        p = params
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        longs = indicators['long_entry'].fillna(False).to_numpy(dtype=bool)
        shorts = indicators['short_entry'].fillna(False).to_numpy(dtype=bool)
        long_sizes = indicators['long_size'].to_numpy(dtype=float)
        short_sizes = indicators['short_size'].to_numpy(dtype=float)
        direction = 0
        entry_price = np.nan
        entry_bar = -1
        held_size = 1.0
        for i in range(n):
            if direction:
                # The preceding close's entry decision fills at this open.
                if not np.isfinite(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        direction = 0
                        continue
                    entry_price = opens[i]
                    entry_bar = i
                gain = direction * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= p.profit_target) or i - entry_bar + 1 >= p.max_hold:
                    # This close's exit decision fills at the following open.
                    direction = 0
                    entry_price = np.nan
                    continue
            else:
                if longs[i]:
                    direction = 1
                    held_size = long_sizes[i]
                elif shorts[i]:
                    direction = -1
                    held_size = short_sizes[i]
                if direction:
                    entry_price = np.nan
                    held_size = float(np.clip(held_size, p.min_size, p.max_size)) if np.isfinite(held_size) else p.min_size
            signal[i] = direction
            if direction:
                size[i] = held_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=5)
    for direction, target, holding, final_close in ((1, 0.02, 10, 103.0), (-1, 0.02, 10, 97.0), (1, 0.5, 2, 100.0)):
        data = pd.DataFrame({'open': [90.0, 100.0, 100.0, 100.0, 100.0],
                             'close': [90.0, 100.0, final_close, final_close, final_close]}, index=index)
        ind = pd.DataFrame({'long_entry': [direction == 1, False, False, False, False],
                            'short_entry': [direction == -1, False, False, False, False],
                            'long_size': 0.625, 'short_size': 0.625}, index=index)
        result = strategy.generate_signals(data, ind, None, ExcursionParams(profit_target=target, max_hold=holding)).data
        assert result['signal'].tolist() == [0, direction, direction, 0, 0]
        assert result['size'].iloc[1:3].eq(0.625).all()
        assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
