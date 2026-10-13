from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_window: int = 40
    fast_window: int = 3
    slow_window: int = 15
    atr_window: int = 14
    min_drawdown: float = 0.02
    compression_limit: float = 0.8
    asymmetry_limit: float = 0.75
    stop_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791245104'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.peak_window, params.fast_window, params.slow_window, params.atr_window) + 1

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (params.peak_window >= 2 and 1 <= params.fast_window < params.slow_window and params.atr_window >= 1):
            raise ValueError('Invalid indicator windows')
        if not (0 < params.min_drawdown < 1 and 0 < params.compression_limit < 1 and 0 < params.asymmetry_limit < 1 and params.stop_atr > 0):
            raise ValueError('Invalid thresholds')
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['peak'] = data['close'].rolling(params.peak_window).max()
        out['drawdown'] = 1.0 - data['close'] / out['peak'].where(out['peak'] > 0)
        out['drawdown_change'] = out['drawdown'].diff()
        out['up_excursion'] = (data['high'] - out['previous_close']).clip(lower=0)
        out['down_excursion'] = (out['previous_close'] - data['low']).clip(lower=0)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        for name in ('up_excursion', 'down_excursion', 'true_range'):
            out[name + '_fast'] = out[name].rolling(params.fast_window).mean()
            out[name + '_slow'] = out[name].rolling(params.slow_window).mean()
            denominator = out[name + '_slow'].where(out[name + '_slow'] > 0)
            out[name + '_ratio'] = out[name + '_fast'] / denominator
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        x = indicators
        ready = np.isfinite(x[['drawdown', 'drawdown_change', 'atr', 'up_excursion_ratio', 'down_excursion_ratio', 'true_range_ratio']]).all(axis=1)
        compressed = ready & (x['atr'] > 0) & (x['true_range_ratio'] < params.compression_limit)
        deep = x['drawdown'] >= params.min_drawdown
        long_entry = compressed & deep & (x['drawdown_change'] < 0) & (x['down_excursion_ratio'] < params.asymmetry_limit * x['up_excursion_ratio'])
        short_entry = compressed & deep & (x['drawdown_change'] > 0) & (x['up_excursion_ratio'] < params.asymmetry_limit * x['down_excursion_ratio'])
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = x['atr'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if position:
                # The previous close's decision fills at this bar's open.
                if i == entry_bar + 1:
                    if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                        position = 0
                        continue
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop) < 0
                # Two held sessions; close-based exits execute next open.
                if stopped or i - entry_bar >= 2 or not np.isfinite(closes[i]):
                    position = 0
                desired[i] = position
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i
                desired[i] = position
        df = pd.DataFrame({'signal': desired, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    x = pd.DataFrame({'drawdown': 0.03, 'drawdown_change': 0.0, 'atr': 2.0, 'up_excursion_ratio': 0.8, 'down_excursion_ratio': 0.3, 'true_range_ratio': 0.6}, index=index)
    x.loc[index[0], 'drawdown_change'] = -0.001
    signals = GeneratedStrategy.generate_signals(None, data, x, None, params).data
    assert signals['signal'].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], 'close'] = 97.0
    x.loc[index[1], 'atr'] = 100.0
    signals = GeneratedStrategy.generate_signals(None, data, x, None, params).data
    assert signals['signal'].tolist() == [0, 1, 0, 0, 0]
    x['up_excursion_ratio'], x['down_excursion_ratio'] = 0.3, 0.8
    x.loc[index[0], 'drawdown_change'] = 0.001
    data.loc[index[1], 'close'] = 103.0
    signals = GeneratedStrategy.generate_signals(None, data, x, None, params).data
    assert signals['signal'].tolist() == [0, -1, 0, 0, 0]
    x['drawdown_change'] = 0.0
    assert not GeneratedStrategy.generate_signals(None, data, x, None, params).data['signal'].any()


if __name__ == '__main__':
    _self_check()
