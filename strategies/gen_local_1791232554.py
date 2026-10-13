from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalRangeParams:
    season_window: int = 126
    range_window: int = 10
    atr_window: int = 14
    contraction: float = 0.85
    seasonal_threshold: float = 0.0003
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8
    hold_bars: int = 2
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[SeasonalRangeParams]):
    strategy_id = 'gen_local_1791232554'

    @classmethod
    def params_type(cls) -> type[SeasonalRangeParams]:
        return SeasonalRangeParams

    @staticmethod
    def warmup_bars(params: SeasonalRangeParams) -> int:
        return int(max(params.season_window + 1, params.range_window + 1, params.atr_window + 1, 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalRangeParams) -> pd.DataFrame:
        if min(params.season_window, params.range_window, params.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not (0.0 < params.contraction <= 1.0 and params.seasonal_threshold >= 0.0):
            raise ValueError('Invalid entry thresholds.')
        if not (params.breakeven_pct > 0.0 and params.trail_atr > 0.0):
            raise ValueError('Exit thresholds must be positive.')
        if params.hold_bars not in (1, 2) or not (0.0 < params.size <= 1.0):
            raise ValueError('Holding period must be one or two bars; size must be positive and at most one.')
        out = pd.DataFrame(index=data.index)
        out['intraday_return'] = data['close'].div(data['open'].where(data['open'] > 0.0)) - 1.0
        out['range'] = (data['high'] - data['low']).where(data['high'] >= data['low'])
        out['midpoint'] = (data['high'] + data['low']) * 0.5
        out['range_baseline'] = out['range'].rolling(params.range_window, min_periods=params.range_window).mean().shift(1)
        out['range_ratio'] = out['range'].div(out['range_baseline'].where(out['range_baseline'] > 0.0))
        out['migration'] = out['midpoint'].diff().div(out['range_baseline'].where(out['range_baseline'] > 0.0))
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['seasonal_edge'] = np.nan
        weekdays = pd.Series(data.index.dayofweek, index=data.index)
        valid_return = np.isfinite(out['intraday_return'])
        for weekday in range(5):
            member = weekdays.eq(weekday)
            observations = member & valid_return
            history = out['intraday_return'].where(observations, 0.0)
            counts = observations.astype(float).rolling(params.season_window, min_periods=params.season_window).sum().shift(1)
            sums = history.rolling(params.season_window, min_periods=params.season_window).sum().shift(1)
            edge = sums.div(counts.where(counts >= 8.0))
            out.loc[member, 'seasonal_edge'] = edge.loc[member]
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalRangeParams) -> SignalFrame:
        edge = indicators['seasonal_edge']
        migration = indicators['migration']
        ready = np.isfinite(indicators[['seasonal_edge', 'migration', 'range_ratio', 'atr']]).all(axis=1)
        ready &= indicators['atr'].gt(0.0) & indicators['range_ratio'].gt(0.0)
        compressed = indicators['range_ratio'].le(params.contraction)
        long_entry = ready & compressed & edge.gt(params.seasonal_threshold) & migration.gt(0.0)
        short_entry = ready & compressed & edge.lt(-params.seasonal_threshold) & migration.lt(0.0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position, pending, age, armed = 0, False, 0, False
        entry, stop = 0.0, 0.0
        for i in range(len(data)):
            opened, high, low, close = prices[i]
            if not np.isfinite(prices[i]).all() or min(prices[i]) <= 0.0 or high < low:
                position, pending = 0, False
                continue
            if position:
                if pending:
                    # The previous close's decision fills at this bar's open.
                    entry = position * opened
                    stop = entry - params.trail_atr * atr[i - 1]
                    pending, age, armed = False, 0, False
                age += 1
                # Favorable-price coordinates make the stop increase for either side.
                favorable = high if position == 1 else -low
                adverse = low if position == 1 else -high
                signed_close = position * close
                exit_now = adverse <= stop
                if not exit_now:
                    armed = armed or favorable >= entry + abs(entry) * params.breakeven_pct
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0.0:
                            stop = max(stop, favorable - params.trail_atr * atr[i])
                        # Only the close tests a newly raised stop: intrabar order is unknown.
                        exit_now = signed_close <= stop
                if exit_now or age >= params.hold_bars:
                    position, pending = 0, False
                    continue
            elif candidates[i] != 0:
                position, pending = int(candidates[i]), True
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': float(params.size)}, index=data.index)
        # Stop breaches are close decisions; liquidation also executes next bar.
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-06', periods=6, freq='B')
    params = SeasonalRangeParams()
    for side in (1, -1):
        data = pd.DataFrame({
            'open': [100.0, 100.0, 100.8, 100.0, 100.0, 100.0],
            'high': [100.5, 101.0, 101.2, 100.5, 100.5, 100.5],
            'low': [99.5, 99.5, 100.0, 99.5, 99.5, 99.5],
            'close': [100.0, 100.8, 100.5, 100.0, 100.0, 100.0],
            'volume': 1000.0
        }, index=index)
        if side == -1:
            old_high = data['high'].copy()
            data['open'] = 200.0 - data['open']
            data['high'] = 200.0 - data['low']
            data['low'] = 200.0 - old_high
            data['close'] = 200.0 - data['close']
        ind = pd.DataFrame({
            'seasonal_edge': [side * 0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
            'migration': float(side), 'range_ratio': 0.5, 'atr': 1.0
        }, index=index)
        result = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert result['signal'].tolist() == [0, side, side, 0, 0, 0]
        ind['seasonal_edge'] = np.nan
        assert GeneratedStrategy.generate_signals(data, ind, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
