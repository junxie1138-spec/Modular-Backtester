from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletReactionParams:
    baseline: int = 20
    energy_burst: float = 1.6
    fine_share: float = 0.60
    volume_ratio: float = 1.1
    profit_target: float = 0.006
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletReactionParams]):
    strategy_id = 'gen_local-w2_1791266712'

    @classmethod
    def params_type(cls):
        return WaveletReactionParams

    def warmup_bars(self, params: WaveletReactionParams) -> int:
        return max(2, int(params.baseline)) + 9

    def indicators(self, data: pd.DataFrame, params: WaveletReactionParams) -> pd.DataFrame:
        window = max(2, int(params.baseline))
        out = pd.DataFrame(index=data.index)
        out['return'] = np.log(data['close'].where(data['close'] > 0)).diff()
        out['haar2'] = (out['return'] - out['return'].shift(1)) / np.sqrt(2.0)
        out['sum2'] = out['return'].rolling(2, min_periods=2).sum()
        out['haar4'] = (out['sum2'] - out['sum2'].shift(2)) / 2.0
        out['sum4'] = out['return'].rolling(4, min_periods=4).sum()
        out['haar8'] = (out['sum4'] - out['sum4'].shift(4)) / np.sqrt(8.0)
        out['energy2'] = out['haar2'].pow(2)
        out['energy4'] = out['haar4'].pow(2)
        out['energy8'] = out['haar8'].pow(2)
        out['total_energy'] = out['energy2'] + out['energy4'] + out['energy8']
        out['fine_fraction'] = out['energy2'] / out['total_energy'].where(out['total_energy'] > 0)
        out['previous_fraction'] = out['fine_fraction'].shift(1)
        out['energy_baseline'] = out['total_energy'].rolling(window, min_periods=window).mean().shift(1)
        out['range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['close_pressure'] = (2.0 * data['close'] - data['high'] - data['low']) / out['range']
        out['body_pressure'] = (data['close'] - data['open']) / out['range']
        out['volume_baseline'] = data['volume'].rolling(window, min_periods=window).mean().shift(1)
        out['relative_volume'] = data['volume'] / out['volume_baseline'].where(out['volume_baseline'] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletReactionParams) -> SignalFrame:
        if params.profit_target <= 0 or params.hold_bars not in (1, 2):
            raise ValueError('profit_target must be positive and hold_bars must be 1 or 2')
        wavelet_change = (
            indicators['fine_fraction'].ge(params.fine_share)
            & indicators['previous_fraction'].lt(params.fine_share)
            & indicators['energy_baseline'].gt(0)
            & indicators['total_energy'].gt(params.energy_burst * indicators['energy_baseline'])
        )
        volume_confirmation = indicators['relative_volume'].ge(params.volume_ratio)
        long_entry = (
            wavelet_change & volume_confirmation
            & indicators['haar2'].gt(0)
            & indicators['return'].gt(0)
            & indicators['close_pressure'].gt(0.25)
            & indicators['body_pressure'].gt(0)
        )
        short_entry = (
            wavelet_change & volume_confirmation
            & indicators['haar2'].lt(0)
            & indicators['return'].lt(0)
            & indicators['close_pressure'].lt(-0.25)
            & indicators['body_pressure'].lt(0)
        )
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        position = 0
            elif entries[i] != 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 98.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    features = pd.DataFrame({
        'fine_fraction': [0.8, 0.2, 0.2, 0.8, 0.2, 0.2],
        'previous_fraction': 0.2,
        'energy_baseline': 1.0,
        'total_energy': 2.0,
        'relative_volume': 2.0,
        'haar2': [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        'return': [0.01, 0.0, 0.0, -0.01, 0.0, 0.0],
        'close_pressure': [0.8, 0.0, 0.0, -0.8, 0.0, 0.0],
        'body_pressure': [0.5, 0.0, 0.0, -0.5, 0.0, 0.0],
    }, index=index)
    strategy = GeneratedStrategy()
    params = WaveletReactionParams()
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1]
    data['close'] = [100.0, 101.0, 100.0, 100.0, 99.0, 100.0]
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0]
    features['relative_volume'] = 0.5
    assert strategy.generate_signals(data, features, None, params).data['signal'].eq(0).all()
    assert strategy.warmup_bars(params) == 29


if __name__ == '__main__':
    _self_check()
