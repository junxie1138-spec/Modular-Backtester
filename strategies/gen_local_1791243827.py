from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OccupancyParams:
    lookback: int = 40
    holding_bars: int = 18

    def __post_init__(self):
        if self.lookback < 10:
            raise ValueError('lookback must be at least 10')
        if not 15 <= self.holding_bars <= 20:
            raise ValueError('holding_bars must be between 15 and 20')


class GeneratedStrategy(BaseStrategy[OccupancyParams]):
    strategy_id = 'gen_local_1791243827'

    @classmethod
    def params_type(cls):
        return OccupancyParams

    def warmup_bars(self, params: OccupancyParams) -> int:
        return 2 * params.lookback + 8

    def indicators(self, data: pd.DataFrame, params: OccupancyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['floor'] = data['low'].rolling(params.lookback).min().shift(1)
        out['ceiling'] = data['high'].rolling(params.lookback).max().shift(1)
        out['width'] = (out['ceiling'] - out['floor']).where(out['ceiling'] > out['floor'])
        out['occupancy'] = ((data['close'] - out['floor']) / out['width']).clip(0.02, 0.98)
        out['log_odds'] = np.log(out['occupancy'] / (1.0 - out['occupancy']))
        out['migration'] = out['log_odds'].diff()
        out['latent_slope'] = out['migration'].rolling(5).mean()
        out['capacity'] = 4.0 * out['occupancy'].shift(1) * (1.0 - out['occupancy'].shift(1))
        out['forecast'] = out['latent_slope'].shift(1) * out['capacity']
        out['noise'] = out['migration'].rolling(params.lookback).std(ddof=0).shift(1) / np.sqrt(5.0)
        out['innovation'] = (out['latent_slope'] - out['forecast']) / out['noise'].where(out['noise'] > 0.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        long_entry = (
            out['occupancy'].between(0.15, 0.50)
            & out['latent_slope'].gt(0.0)
            & out['innovation'].gt(0.75)
        )
        short_entry = (
            out['occupancy'].between(0.50, 0.85)
            & out['latent_slope'].lt(0.0)
            & out['innovation'].lt(-0.75)
        )
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OccupancyParams) -> SignalFrame:
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, candidate in enumerate(entries):
            if position != 0:
                if bar == exit_bar:
                    position = 0
                signal[bar] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = bar + params.holding_bars
            signal[bar] = position
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = OccupancyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=100, freq='B')
    close = 100.0 + np.arange(100) * 0.05 + np.sin(np.arange(100) / 4.0)
    data = pd.DataFrame({'open': close, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['entry'].iloc[:params.lookback + 1].eq(0).all()
    prefix = strategy.indicators(data.iloc[:80], params)
    pd.testing.assert_frame_equal(derived.iloc[:80], prefix)
    for direction in (1, -1):
        entries = pd.DataFrame({'entry': direction}, index=index)
        result = strategy.generate_signals(data, entries, None, params).data
        assert result['signal'].iloc[0] == 0
        assert result['signal'].iloc[1:params.holding_bars + 1].eq(direction).all()
        assert result['signal'].iloc[params.holding_bars + 1] == 0
        assert result['signal'].iloc[params.holding_bars + 2] == direction
        assert result['size'].gt(0.0).all()
