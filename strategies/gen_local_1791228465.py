from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    complexity_window: int = 20
    seasonal_window: int = 504
    complexity_split: float = 0.85
    min_samples: int = 10
    edge_threshold: float = 0.003
    profit_target: float = 0.02
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791228465'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.seasonal_window + params.complexity_window + params.hold_days + 3))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (params.complexity_window >= 3 and params.seasonal_window >= 20 and params.min_samples >= 1 and 3 <= params.hold_days <= 5 and 0 < params.complexity_split < 1 and params.edge_threshold > 0 and params.profit_target > 0):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        triples = np.column_stack([out['return'].shift(2), out['return'].shift(1), out['return']])
        order = np.argsort(triples, axis=1, kind='stable')
        code = pd.Series(9 * order[:, 0] + 3 * order[:, 1] + order[:, 2], index=data.index, dtype=float)
        out['ordinal_code'] = code.where(np.isfinite(triples).all(axis=1))
        entropy = pd.Series(0.0, index=data.index)
        for pattern in (5, 7, 11, 15, 19, 21):
            membership = out['ordinal_code'].eq(pattern).astype(float).where(out['ordinal_code'].notna())
            frequency = membership.rolling(params.complexity_window, min_periods=params.complexity_window).mean()
            out[f'pattern_frequency_{pattern}'] = frequency
            entropy = entropy - frequency * np.log(frequency.where(frequency > 0, 1.0))
        out['complexity'] = entropy / np.log(6.0)
        weekday = pd.Series(data.index.dayofweek, index=data.index, dtype=float)
        out['weekday'] = weekday
        out['bucket'] = (weekday + 5 * out['complexity'].ge(params.complexity_split).astype(int)).where(out['complexity'].notna() & weekday.lt(5))
        out['matured_bucket'] = out['bucket'].shift(params.hold_days)
        out['matured_return'] = close.pct_change(params.hold_days, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['seasonal_edge'] = np.nan
        out['seasonal_count'] = 0.0
        for bucket in range(10):
            eligible = out['matured_bucket'].eq(bucket) & out['matured_return'].notna()
            count = eligible.astype(float).rolling(params.seasonal_window, min_periods=params.seasonal_window).sum()
            total = out['matured_return'].where(eligible, 0.0).rolling(params.seasonal_window, min_periods=params.seasonal_window).sum()
            edge = total / count.where(count > 0)
            out[f'bucket_count_{bucket}'] = count
            out[f'bucket_edge_{bucket}'] = edge
            current = out['bucket'].eq(bucket)
            out.loc[current, 'seasonal_count'] = count.loc[current]
            out.loc[current, 'seasonal_edge'] = edge.loc[current]
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        average = indicators['ma200'].to_numpy(dtype=float)
        edge = indicators['seasonal_edge'].to_numpy(dtype=float)
        count = indicators['seasonal_count'].to_numpy(dtype=float)
        ready = np.isfinite(close) & (close > 0) & np.isfinite(average) & np.isfinite(edge) & (count >= params.min_samples)
        candidates = np.zeros(len(data), dtype=int)
        candidates[ready & (close > average) & (edge >= params.edge_threshold)] = 1
        candidates[ready & (close < average) & (edge <= -params.edge_threshold)] = -1
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if entry_bar < 0:
                    entry_price = opening[i]
                    entry_bar = i
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(close[i]) and close[i] > 0
                gain = position * (close[i] / entry_price - 1.0) if valid else 0.0
                if not valid or gain >= params.profit_target or i - entry_bar + 1 >= params.hold_days:
                    position = 0
                    entry_bar = -1
                    entry_price = np.nan
            elif candidates[i]:
                position = int(candidates[i])
                entry_bar = -1
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=12, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    data.loc[index[2], 'close'] = 103.0
    derived = pd.DataFrame({'ma200': 100.0, 'seasonal_edge': 0.0, 'seasonal_count': 20.0}, index=index)
    derived.loc[index[0], ['ma200', 'seasonal_edge']] = [90.0, 0.01]
    derived.loc[index[6], ['ma200', 'seasonal_edge']] = [110.0, -0.01]
    result = GeneratedStrategy().generate_signals(data, derived, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 0]
    assert result['size'].gt(0).all()
