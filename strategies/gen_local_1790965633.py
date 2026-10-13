from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    position_window: int = 20
    roc_window: int = 3
    atr_window: int = 14
    edge: float = 0.30
    acceleration_floor: float = 0.0004
    atr_multiple: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790965633"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.position_window, params.roc_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.position_window, params.roc_window, params.atr_window, params.holding_bars) < 1:
            raise ValueError("Window lengths and holding_bars must be positive")
        if not 0.0 < params.edge < 0.5:
            raise ValueError("edge must be between zero and one-half")
        if not np.isfinite(params.acceleration_floor) or params.acceleration_floor < 0.0:
            raise ValueError("acceleration_floor must be finite and nonnegative")
        if not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0.0:
            raise ValueError("atr_multiple must be finite and positive")
        close = data['close'].astype(float)
        previous_close = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['range_low'] = close.rolling(params.position_window, min_periods=params.position_window).min()
        out['range_high'] = close.rolling(params.position_window, min_periods=params.position_window).max()
        out['range_width'] = (out['range_high'] - out['range_low']).replace(0.0, np.nan)
        out['relative_position'] = (close - out['range_low']) / out['range_width']
        out['roc'] = close / close.shift(params.roc_window).replace(0.0, np.nan) - 1.0
        out['acceleration'] = out['roc'].diff()
        out['previous_acceleration'] = out['acceleration'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        valid = np.isfinite(out[['ma200', 'relative_position', 'roc', 'acceleration', 'previous_acceleration', 'atr']]).all(axis=1) & (out['atr'] > 0.0) & np.isfinite(close) & (close > 0.0)
        out['long_entry'] = valid & (close > out['ma200']) & (out['relative_position'] <= params.edge) & (out['roc'] < 0.0) & (out['previous_acceleration'] <= params.acceleration_floor) & (out['acceleration'] > params.acceleration_floor)
        out['short_entry'] = valid & (close < out['ma200']) & (out['relative_position'] >= 1.0 - params.edge) & (out['roc'] > 0.0) & (out['previous_acceleration'] >= -params.acceleration_floor) & (out['acceleration'] < -params.acceleration_floor)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = -np.inf
        low_water = np.inf
        trailing_floor = -np.inf
        short_ceiling = np.inf
        columns = ['atr', 'ma200', 'long_entry', 'short_entry']
        values = indicators[columns].to_numpy()
        for i, close in enumerate(data['close'].to_numpy(dtype=float)):
            atr, ma200, long_entry, short_entry = values[i]
            if not (np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0 and np.isfinite(ma200)):
                position = 0
                decisions[i] = 0
                continue
            if position:
                # The entry decision filled at this bar's open. Track closes starting here.
                age += 1
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                trailing_floor = max(trailing_floor, high_water - params.atr_multiple * atr)
                short_ceiling = min(short_ceiling, low_water + params.atr_multiple * atr)
                # Both directions obey the required highest-close trailing floor.
                # Shorts additionally have a mirrored protective trailing ceiling.
                stopped = close <= trailing_floor or (position == -1 and close >= short_ceiling)
                regime_exit = (position == 1 and close <= ma200) or (position == -1 and close >= ma200)
                if stopped or regime_exit or age >= params.holding_bars:
                    position = 0
                decisions[i] = position
                continue
            if long_entry == True:
                position = 1
            elif short_entry == True:
                position = -1
            if position:
                age = 0
                high_water = -np.inf
                low_water = np.inf
                trailing_floor = -np.inf
                short_ceiling = np.inf
            decisions[i] = position
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(atr_multiple=2.0)
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [99.0, 100.0, 110.0, 107.0, 106.0]}, index=index)
    indicators = pd.DataFrame({'atr': [1.0, 1.0, 1.0, 10.0, 1.0], 'ma200': 90.0, 'long_entry': [True, False, False, False, False], 'short_entry': False}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    assert (result['size'] > 0.0).all()
    data['close'] = [100.0, 100.0, 99.0, 96.0, 95.0]
    indicators['ma200'] = 120.0
    indicators['atr'] = 1.0
    indicators['long_entry'] = False
    indicators['short_entry'] = [True, False, False, False, False]
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, -1, -1, -1, 0]


if __name__ == '__main__':
    _self_check()
