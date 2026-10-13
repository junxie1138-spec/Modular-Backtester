from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    max_distance: float = 0.5
    volume_ratio: float = 1.15
    min_move: float = 0.002
    profit_target: float = 0.012
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791312810'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return 9

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not (0.0 <= params.max_distance <= 2.0):
            raise ValueError('max_distance must be between zero and two')
        if params.volume_ratio <= 0.0 or params.min_move < 0.0:
            raise ValueError('Invalid volume or move threshold')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        out['return'] = close.div(close.shift(1)).sub(1.0)
        out['path_amplitude'] = out['return'].abs().rolling(3, min_periods=3).sum()
        denominator = out['path_amplitude'].replace(0.0, np.nan)
        for component in range(3):
            out[f'shape_{component}'] = out['return'].shift(component).div(denominator)
        for lag in (4, 5):
            differences = pd.concat([
                (out[f'shape_{component}'] - out[f'shape_{component}'].shift(lag)).abs()
                for component in range(3)
            ], axis=1)
            out[f'distance_{lag}'] = differences.sum(axis=1, min_count=3)
            out[f'outcome_{lag}'] = out['return'].shift(lag - 1)
        choose_four = out['distance_4'] <= out['distance_5']
        out['match_distance'] = out['distance_4'].where(choose_four, out['distance_5'])
        out['forecast'] = out['outcome_4'].where(choose_four, out['outcome_5'])
        out['volume_baseline'] = data['volume'].shift(1).rolling(5, min_periods=5).mean()
        out['relative_volume'] = data['volume'].div(out['volume_baseline'].where(out['volume_baseline'] > 0.0))
        ready = out[['distance_4', 'distance_5', 'forecast', 'relative_volume', 'return']].notna().all(axis=1)
        confirmed = (
            ready
            & (out['match_distance'] <= params.max_distance)
            & (out['relative_volume'] >= params.volume_ratio)
            & (out['return'].abs() >= params.min_move)
            & (out['return'] * out['forecast'] > 0.0)
        )
        out['entry_direction'] = np.sign(out['forecast']).where(confirmed, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        if params.hold_bars not in (3, 4, 5) or params.profit_target <= 0.0:
            raise ValueError('Use a positive target and a three-to-five-session hold')
        candidates = indicators['entry_direction'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        age = 0
        pending_entry = False
        for i in range(len(data)):
            if position:
                if pending_entry:
                    entry_price = opens[i]
                    pending_entry = False
                age += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                if not valid_entry or gain >= params.profit_target or age >= params.hold_bars:
                    position = 0
                    age = 0
                    entry_price = np.nan
            elif candidates[i] in (-1, 1):
                position = int(candidates[i])
                pending_entry = True
                age = 0
            desired[i] = position
        df = pd.DataFrame({'signal': desired, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2024-01-01', periods=7, freq='D')
    sample = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    entries = pd.DataFrame({'entry_direction': [1, 0, 0, 0, 0, 0, 0]}, index=index)
    params = AnalogParams()
    timed = GeneratedStrategy.generate_signals(sample, entries, None, params).data
    assert timed['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    sample.loc[index[3], 'close'] = 102.0
    targeted = GeneratedStrategy.generate_signals(sample, entries, None, params).data
    assert targeted['signal'].tolist() == [0, 1, 1, 1, 0, 0, 0]
    entries['entry_direction'] = [-1, 0, 0, 0, 0, 0, 0]
    sample.loc[index[3], 'close'] = 98.0
    short = GeneratedStrategy.generate_signals(sample, entries, None, params).data
    assert short['signal'].tolist() == [0, -1, -1, -1, 0, 0, 0]
