from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DepthParams:
    depth_window: int = 60
    profit_target: float = 0.025


class GeneratedStrategy(BaseStrategy[DepthParams]):
    strategy_id = "gen_local_1791247935"

    @classmethod
    def params_type(cls):
        return DepthParams

    @staticmethod
    def warmup_bars(params: DepthParams) -> int:
        return max(2, int(params.depth_window)) + 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: DepthParams) -> pd.DataFrame:
        window = max(2, int(params.depth_window))
        result = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        result['peak'] = data['high'].where(data['high'] > 0).rolling(window, min_periods=window).max()
        result['trough'] = data['low'].where(data['low'] > 0).rolling(window, min_periods=window).min()
        result['down_depth'] = np.log(result['peak'] / close).clip(lower=0)
        result['up_depth'] = np.log(close / result['trough']).clip(lower=0)
        result['down_energy'] = result['down_depth'].pow(2).rolling(10, min_periods=10).mean()
        result['up_energy'] = result['up_depth'].pow(2).rolling(10, min_periods=10).mean()
        result['total_energy'] = result['up_energy'] + result['down_energy']
        result['polarization'] = ((result['up_energy'] - result['down_energy']) / result['total_energy'].where(result['total_energy'] > 0)).replace([np.inf, -np.inf], np.nan)
        return result

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DepthParams) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('profit_target must be positive and finite')
        strength = indicators['polarization']
        previous = strength.shift(1)
        entries = pd.Series(0, index=data.index, dtype=int)
        entries.loc[(strength > 0.65) & (previous <= 0.65)] = 1
        entries.loc[(strength < -0.65) & (previous >= -0.65)] = -1
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        candidates = entries.to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid else 0.0
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= 10:
                    position = 0
                    entry_price = np.nan
                    continue
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_bar = i + 1
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=14)
    for direction, exit_close, expected_exit in [(1, 103.0, 3), (-1, 97.0, 3), (1, 100.0, 12), (-1, 100.0, 12)]:
        sample = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 100.0, 'volume': 1000.0}, index=index)
        sample.loc[index[2], 'close'] = exit_close
        strength = pd.DataFrame({'polarization': [0.0] + [0.8 * direction] * 13}, index=index)
        output = GeneratedStrategy.generate_signals(sample, strength, None, DepthParams()).data
        assert output['signal'].iloc[:2].eq(0).all()
        assert output['signal'].iloc[2:expected_exit].eq(direction).all()
        assert output['signal'].iloc[expected_exit:].eq(0).all()
        assert output['size'].gt(0).all()
