from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_bars: int = 3
    flow_bars: int = 5
    regime_bars: int = 20
    overlap_cutoff: float = 0.60
    acceleration_min: float = 0.003
    profit_target: float = 0.025
    max_hold_bars: int = 8

    def __post_init__(self):
        if min(self.roc_bars, self.flow_bars, self.regime_bars, self.max_hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.0 < self.overlap_cutoff < 1.0:
            raise ValueError('overlap_cutoff must be between zero and one.')
        if not np.isfinite(self.acceleration_min) or self.acceleration_min <= 0:
            raise ValueError('acceleration_min must be finite and positive.')
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive.')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local-w2_1791341259'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(2 * params.roc_bars + 1, params.flow_bars + 1, params.regime_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['roc'] = data['close'].pct_change(params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff(params.roc_bars)
        out['range'] = data['high'] - data['low']
        out['close_location'] = ((2.0 * data['close'] - data['high'] - data['low']) / out['range'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out.loc[out['range'].eq(0.0), 'close_location'] = 0.0
        out['signed_volume'] = out['close_location'] * data['volume']
        out['flow_sum'] = out['signed_volume'].rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        out['volume_sum'] = data['volume'].rolling(params.flow_bars, min_periods=params.flow_bars).sum()
        out['pressure'] = out['flow_sum'] / out['volume_sum'].where(out['volume_sum'] > 0.0)
        out['pressure_change'] = out['pressure'].diff()
        out['intersection'] = (np.minimum(data['high'], data['high'].shift(1)) - np.maximum(data['low'], data['low'].shift(1))).clip(lower=0.0)
        out['union'] = np.maximum(data['high'], data['high'].shift(1)) - np.minimum(data['low'], data['low'].shift(1))
        out['overlap_ratio'] = out['intersection'] / out['union'].where(out['union'] > 0.0)
        out['overlap'] = out['overlap_ratio'].rolling(params.regime_bars, min_periods=params.regime_bars).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = indicators[['acceleration', 'pressure', 'pressure_change', 'overlap']].notna().all(axis=1)
        rotation = indicators['overlap'] >= params.overlap_cutoff
        pressure = indicators['pressure']
        change = indicators['pressure_change']
        upward_flow = (change > 0.0) & ((~rotation & (pressure > 0.10)) | (rotation & (pressure < -0.10)))
        downward_flow = (change < 0.0) & ((~rotation & (pressure < -0.10)) | (rotation & (pressure > 0.10)))
        long_entry = ready & (indicators['acceleration'] > params.acceleration_min) & upward_flow
        short_entry = ready & (indicators['acceleration'] < -params.acceleration_min) & downward_flow
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if position:
                # A decision on the previous close becomes an entry at this open.
                if i == entry_bar + 1:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and np.isfinite(closes[i]) else np.nan
                if not valid_entry or gain >= params.profit_target or i - entry_bar >= params.max_hold_bars:
                    position = 0
            elif candidates[i]:
                position = int(candidates[i])
                entry_bar = i
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'acceleration': 0.01, 'pressure': 0.20, 'pressure_change': 0.01, 'overlap': 0.20}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    signal = strategy.generate_signals(data, ind, ctx, params).data['signal']
    assert signal.iloc[0] == 0 and signal.iloc[1] == 1
    assert signal.iloc[8] == 1 and signal.iloc[9] == 0
    data.loc[index[1], 'close'] = 103.0
    signal = strategy.generate_signals(data, ind, ctx, params).data['signal']
    assert signal.iloc[1] == 1 and signal.iloc[2] == 0
    ind['acceleration'] = -0.01
    ind['pressure'] = -0.20
    ind['pressure_change'] = -0.01
    data.loc[index[1], 'close'] = 97.0
    signal = strategy.generate_signals(data, ind, ctx, params).data['signal']
    assert signal.iloc[1] == -1 and signal.iloc[2] == 0
    ind['pressure_change'] = 0.01
    assert strategy.generate_signals(data, ind, ctx, params).data['signal'].eq(0).all()
    ind['acceleration'] = np.nan
    assert strategy.generate_signals(data, ind, ctx, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
