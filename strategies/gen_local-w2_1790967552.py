from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OccupancyParams:
    peak_window: int = 63
    history_window: int = 126
    occupancy_window: int = 10
    tail_quantile: float = 0.80
    compression_quantile: float = 0.30
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[OccupancyParams]):
    strategy_id = 'gen_local-w2_1790967552'

    @classmethod
    def params_type(cls) -> type[OccupancyParams]:
        return OccupancyParams

    def warmup_bars(self, params: OccupancyParams) -> int:
        return int(params.peak_window + 2 * params.history_window + params.occupancy_window + 2)

    def indicators(self, data: pd.DataFrame, params: OccupancyParams) -> pd.DataFrame:
        p = params
        if min(p.peak_window, p.history_window, p.occupancy_window, p.hold_bars) < 1:
            raise ValueError('Window lengths and holding period must be positive.')
        if not (0.5 < p.tail_quantile < 1.0 and 0.0 < p.compression_quantile < 1.0):
            raise ValueError('Invalid percentile parameters.')
        if not np.isfinite(p.profit_target) or p.profit_target <= 0.0:
            raise ValueError('Profit target must be positive and finite.')
        out = pd.DataFrame(index=data.index)
        out['peak'] = data['high'].rolling(p.peak_window, min_periods=p.peak_window).max()
        out['drawdown'] = 1.0 - data['close'].div(out['peak'].where(out['peak'] > 0.0))
        out['deep_threshold'] = out['drawdown'].shift(1).rolling(p.history_window, min_periods=p.history_window).quantile(p.tail_quantile)
        out['shallow_threshold'] = out['drawdown'].shift(1).rolling(p.history_window, min_periods=p.history_window).quantile(1.0 - p.tail_quantile)
        out['tail_vote'] = ((out['drawdown'] > out['deep_threshold']).astype(float) - (out['drawdown'] < out['shallow_threshold']).astype(float)).where(out[['drawdown', 'deep_threshold', 'shallow_threshold']].notna().all(axis=1))
        out['occupancy'] = out['tail_vote'].rolling(p.occupancy_window, min_periods=p.occupancy_window).mean()
        out['occupancy_upper'] = out['occupancy'].shift(1).rolling(p.history_window, min_periods=p.history_window).quantile(p.tail_quantile)
        out['occupancy_lower'] = out['occupancy'].shift(1).rolling(p.history_window, min_periods=p.history_window).quantile(1.0 - p.tail_quantile)
        out['range_high'] = data['high'].rolling(p.occupancy_window, min_periods=p.occupancy_window).max()
        out['range_low'] = data['low'].rolling(p.occupancy_window, min_periods=p.occupancy_window).min()
        out['range_width'] = (out['range_high'] - out['range_low']).div(data['close'].where(data['close'] > 0.0))
        out['compression_threshold'] = out['range_width'].shift(1).rolling(p.history_window, min_periods=p.history_window).quantile(p.compression_quantile)
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1) & np.isfinite(data['close']) & (data['close'] > 0.0)
        compressed = out['range_width'] < out['compression_threshold']
        long_entry = valid & compressed & (out['occupancy'] > out['occupancy_upper']) & (out['occupancy'] > 0.0)
        short_entry = valid & compressed & (out['occupancy'] < out['occupancy_lower']) & (out['occupancy'] < 0.0)
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OccupancyParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position != 0:
                # The preceding close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    target_hit = np.isfinite(gain) and gain >= params.profit_target
                    expired = i - entry_bar + 1 >= params.hold_bars
                    if target_hit or expired:
                        position = 0
                # An exit decision becomes flat at the following open.
                decisions[i] = position
                continue
            if entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    for side in (1, -1):
        for hit_target in (False, True):
            data = pd.DataFrame({'open': 100.0, 'high': 110.0, 'low': 90.0, 'close': 100.0, 'volume': 1000.0}, index=index)
            if hit_target:
                data.loc[index[2], 'close'] = 100.0 * (1.0 + side * 0.05)
            indicators = pd.DataFrame({'entry': [side, 0, 0, 0, 0, 0]}, index=index)
            result = GeneratedStrategy.generate_signals(None, data, indicators, None, OccupancyParams(hold_bars=3))
            expected = [0, side, side, 0, 0, 0] if hit_target else [0, side, side, side, 0, 0]
            assert result.data['signal'].tolist() == expected
            assert (result.data['size'] > 0.0).all()


if __name__ == '__main__':
    _self_check()
