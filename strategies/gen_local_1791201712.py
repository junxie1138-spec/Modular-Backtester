from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    fractional_order: float = 0.4
    fractional_window: int = 32
    percentile_window: int = 126
    tail_percentile: float = 0.8
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = 'gen_local_1791201712'

    @classmethod
    def params_type(cls) -> type[GapMemoryParams]:
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return int(max(params.fractional_window + params.percentile_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError('fractional_order must be between zero and one')
        if params.fractional_window < 2 or params.percentile_window < 2 or params.atr_window < 1:
            raise ValueError('Lookback windows must be positive; fractional and percentile windows must exceed one')
        if not (0.5 < params.tail_percentile < 1.0):
            raise ValueError('tail_percentile must be between 0.5 and 1.0')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0 or not (3 <= params.holding_bars <= 5):
            raise ValueError('Require a positive finite stop_atr and a three-to-five-session holding period')

        close = data['close'].astype(float).where(lambda x: np.isfinite(x) & (x > 0.0))
        previous = close.shift(1)
        weights = np.empty(params.fractional_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1.0) / lag

        fractional = close * weights[0]
        for lag in range(1, params.fractional_window):
            fractional = fractional + weights[lag] * close.shift(lag)

        history = fractional.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        valid_range = np.isfinite(data['high']) & np.isfinite(data['low']) & previous.notna() & (data['high'] >= data['low'])
        true_range = true_range.where(valid_range)

        result = pd.DataFrame(index=data.index)
        result['fractional_close'] = fractional
        result['upper_percentile'] = history.quantile(params.tail_percentile)
        result['lower_percentile'] = history.quantile(1.0 - params.tail_percentile)
        result['gap'] = data['open'] - previous
        result['repair'] = close - previous
        result['atr'] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapMemoryParams) -> SignalFrame:
        valid = indicators.notna().all(axis=1) & (indicators['atr'] > 0.0)
        long_entry = valid & (indicators['fractional_close'] > indicators['upper_percentile']) & (indicators['gap'] < 0.0) & (indicators['repair'] > 0.0)
        short_entry = valid & (indicators['fractional_close'] < indicators['lower_percentile']) & (indicators['gap'] > 0.0) & (indicators['repair'] < 0.0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        held = 0
        age = 0
        entry_price = np.nan
        stop_distance = np.nan

        for i in range(len(data)):
            # Previous close's target fills at this open; the stop stays fixed.
            target = int(decisions[i - 1]) if i else 0
            if target != held:
                held = target
                age = 0
                if held:
                    entry_price = opens[i]
                    stop_distance = params.stop_atr * atr[i - 1]
                    if not (np.isfinite(entry_price) and entry_price > 0.0 and np.isfinite(stop_distance) and stop_distance > 0.0):
                        held = 0
                        continue

            if held:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0.0:
                    decisions[i] = 0
                    continue
                stopped = (held == 1 and closes[i] < entry_price - stop_distance) or (held == -1 and closes[i] > entry_price + stop_distance)
                decisions[i] = 0 if stopped or age >= params.holding_bars else held
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                decisions[i] = int(candidates[i])

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=11, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': [100.0, 97.0, 100.0, 100.0, 104.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'fractional_close': 0.0, 'upper_percentile': 1.0, 'lower_percentile': -1.0, 'gap': 0.0, 'repair': 0.0, 'atr': 1.0}, index=index)
    indicators.loc[index[[0, 5]], ['fractional_close', 'gap', 'repair']] = [2.0, -1.0, 1.0]
    indicators.loc[index[3], ['fractional_close', 'gap', 'repair']] = [-2.0, 1.0, -1.0]
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, GapMemoryParams()).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0, 1, 1, 1, 1, 0]
    assert (result['size'] > 0.0).all()
    assert strategy.warmup_bars(GapMemoryParams()) == 158


if __name__ == '__main__':
    _self_check()
