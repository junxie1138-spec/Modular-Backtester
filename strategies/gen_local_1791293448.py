from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeContagionParams:
    frontier_window: int = 3
    volatility_window: int = 20
    reproduction_threshold: float = 0.8
    directional_threshold: float = 0.45
    hold_bars: int = 4
    target_annual_vol: float = 0.12
    max_size: float = 0.95


def _fixed_hold(entries, sizes, hold_bars):
    signals = np.zeros(len(entries), dtype=int)
    exposure = np.ones(len(entries), dtype=float)
    side, exit_bar, entry_size = 0, -1, 1.0
    for i in range(len(entries)):
        if side and i == exit_bar:
            side = 0
        elif not side and entries[i]:
            side = int(entries[i])
            exit_bar = i + hold_bars
            entry_size = float(sizes[i])
        signals[i] = side
        if side:
            exposure[i] = entry_size
    return signals, exposure


class GeneratedStrategy(BaseStrategy[RangeContagionParams]):
    strategy_id = 'gen_local_1791293448'

    @classmethod
    def params_type(cls):
        return RangeContagionParams

    @staticmethod
    def warmup_bars(params):
        if params.frontier_window < 1 or params.volatility_window < 2:
            raise ValueError('Lookback windows must be positive; volatility_window must be at least two.')
        if not 3 <= params.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five.')
        if not np.isfinite(params.reproduction_threshold) or params.reproduction_threshold <= 0:
            raise ValueError('reproduction_threshold must be finite and positive.')
        if not np.isfinite(params.directional_threshold) or not 0 < params.directional_threshold < 1:
            raise ValueError('directional_threshold must be between zero and one.')
        if not np.isfinite(params.target_annual_vol) or params.target_annual_vol <= 0:
            raise ValueError('target_annual_vol must be finite and positive.')
        if not np.isfinite(params.max_size) or not 0 < params.max_size <= 1:
            raise ValueError('max_size must be between zero and one.')
        return int(max(params.frontier_window + 2, params.volatility_window + 1))

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        valid = np.isfinite(high) & np.isfinite(low) & (low > 0) & (high >= low)
        high = high.where(valid)
        low = low.where(valid)
        out['upper_frontier'] = (high - high.shift(1)).clip(lower=0)
        out['lower_frontier'] = (low.shift(1) - low).clip(lower=0)
        out['overlap'] = (
            pd.concat([high, high.shift(1)], axis=1).min(axis=1, skipna=False)
            - pd.concat([low, low.shift(1)], axis=1).max(axis=1, skipna=False)
        ).clip(lower=0)
        out['range'] = high - low
        window = params.frontier_window
        out['upper_sum'] = out['upper_frontier'].rolling(window, min_periods=window).sum()
        out['lower_sum'] = out['lower_frontier'].rolling(window, min_periods=window).sum()
        out['overlap_sum'] = out['overlap'].rolling(window, min_periods=window).sum()
        out['range_sum'] = out['range'].rolling(window, min_periods=window).sum()
        out['frontier_sum'] = out['upper_sum'] + out['lower_sum']
        out['shared_core'] = out['overlap_sum'].clip(lower=0.1 * out['range_sum']).replace(0, np.nan)
        out['reproduction'] = out['frontier_sum'] / out['shared_core']
        out['directional_pressure'] = (
            (out['upper_sum'] - out['lower_sum'])
            / out['frontier_sum'].replace(0, np.nan)
        )
        out['log_range_squared'] = np.log(high / low).pow(2)
        out['annual_range_vol'] = np.sqrt(
            out['log_range_squared'].rolling(
                params.volatility_window, min_periods=params.volatility_window
            ).mean() * 252.0 / (4.0 * np.log(2.0))
        )
        out['target_size'] = (
            params.target_annual_vol / out['annual_range_vol'].replace(0, np.nan)
        ).clip(lower=0.01 * params.max_size, upper=params.max_size)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        warmup = self.warmup_bars(params)
        ready = indicators[['reproduction', 'directional_pressure', 'target_size']].notna().all(axis=1)
        ready &= np.arange(len(data)) >= warmup - 1
        spreading = ready & (indicators['reproduction'] > params.reproduction_threshold)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[spreading & (indicators['directional_pressure'] > params.directional_threshold)] = 1
        candidate.loc[spreading & (indicators['directional_pressure'] < -params.directional_threshold)] = -1
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
        sizes = indicators['target_size'].fillna(params.max_size).to_numpy(dtype=float)
        signal, size = _fixed_hold(confirmed.to_numpy(dtype=int), sizes, params.hold_bars)
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.max_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    candidate = pd.Series([1, 1, 0, -1, -1, -1, -1, -1, -1, 0], dtype=int)
    confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0)
    signal, size = _fixed_hold(
        confirmed.to_numpy(dtype=int), np.arange(10) / 100.0 + 0.2, 4
    )
    assert signal.tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1]
    assert np.allclose(size[1:5], 0.21)
    assert np.allclose(size[6:10], 0.26)
    shifted = pd.Series(signal).shift(1).fillna(0).astype(int)
    assert shifted.tolist() == [0, 0, 1, 1, 1, 1, 0, -1, -1, -1]
