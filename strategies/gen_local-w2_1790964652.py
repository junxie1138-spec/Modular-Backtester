from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalVolatilityParams:
    fast_window: int = 5
    slow_window: int = 20
    atr_window: int = 14
    volatility_ratio: float = 1.15
    trail_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SeasonalVolatilityParams]):
    strategy_id = 'gen_local-w2_1790964652'

    @classmethod
    def params_type(cls) -> type[SeasonalVolatilityParams]:
        return SeasonalVolatilityParams

    @staticmethod
    def warmup_bars(params: SeasonalVolatilityParams) -> int:
        return int(max(200, params.fast_window + 1, params.slow_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalVolatilityParams) -> pd.DataFrame:
        if min(params.fast_window, params.slow_window) < 2 or params.atr_window < 1 or params.hold_bars < 1:
            raise ValueError('Invalid lookback or holding period')
        if not np.isfinite(params.volatility_ratio) or params.volatility_ratio <= 0:
            raise ValueError('volatility_ratio must be positive and finite')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be positive and finite')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['fast_vol'] = out['return'].rolling(params.fast_window, min_periods=params.fast_window).std()
        out['slow_vol'] = out['return'].rolling(params.slow_window, min_periods=params.slow_window).std()
        out['volatility_ratio'] = out['fast_vol'].div(out['slow_vol'].replace(0.0, np.nan))
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['month_turn'] = (data.index.day >= 25) | (data.index.day <= 3)
        out['midmonth'] = (data.index.day >= 10) & (data.index.day <= 16)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalVolatilityParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        ratio = indicators['volatility_ratio'].to_numpy(dtype=float)
        returns = indicators['return'].to_numpy(dtype=float)
        turn = indicators['month_turn'].to_numpy(dtype=bool)
        middle = indicators['midmonth'].to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        trailing_floor = -np.inf
        short_ceiling = np.inf
        used_windows = set()

        for i, timestamp in enumerate(data.index):
            valid = np.isfinite([close[i], opening[i], atr[i], ma[i], ratio[i], returns[i]]).all() and atr[i] > 0 and close[i] > 0 and opening[i] > 0
            if not valid:
                position = 0
                age = 0
                continue

            if position:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    high_water = close[i]
                    if position == -1:
                        short_ceiling = opening[i] + params.trail_atr * atr[i]
                age += 1
                high_water = max(high_water, close[i])
                trailing_floor = max(trailing_floor, high_water - params.trail_atr * atr[i])
                regime_exit = (position == 1 and close[i] <= ma[i]) or (position == -1 and close[i] >= ma[i])
                # The mandated high-water drawdown exits both directions.
                # For shorts it takes profit; the separate ceiling limits rallies.
                stop_exit = close[i] < trailing_floor or (position == -1 and close[i] > short_ceiling)
                if stop_exit or regime_exit or age >= params.hold_bars:
                    position = 0
                    age = 0
                signals[i] = position
                continue

            direction = 0
            if ratio[i] >= params.volatility_ratio:
                if turn[i] and close[i] > ma[i] and returns[i] > 0:
                    direction = 1
                elif middle[i] and close[i] < ma[i] and returns[i] < 0:
                    direction = -1
            if direction:
                # Late-month and early-next-month dates share one flow window.
                period = timestamp.to_period('M')
                if direction == 1 and timestamp.day >= 25:
                    period += 1
                key = (period, direction)
                if key not in used_windows:
                    used_windows.add(key)
                    position = direction
                    age = 0
                    high_water = np.nan
                    trailing_floor = -np.inf
                    short_ceiling = np.inf
            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2024-01-02', periods=6, freq='B')
    prices = np.array([100.0, 101.0, 105.0, 104.0, 102.0, 101.0])
    data = pd.DataFrame({'open': prices, 'high': prices + 1, 'low': prices - 1, 'close': prices, 'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'atr': [1.0, 1.0, 1.0, 10.0, 10.0, 10.0], 'ma200': 90.0, 'volatility_ratio': 2.0, 'return': 0.01, 'month_turn': True, 'midmonth': False}, index=index)
    params = SeasonalVolatilityParams(trail_atr=2.0)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200
    # Falling closes also trigger the required high-water exit for shorts.
    data['close'] = [100.0, 99.0, 96.0, 95.0, 94.0, 93.0]
    data['open'] = data['close']
    derived['atr'] = 1.0
    derived['ma200'] = 200.0
    derived['return'] = -0.01
    derived['month_turn'] = False
    derived['midmonth'] = True
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result['signal'].tolist() == [0, -1, -1, -1, 0, 0]


if __name__ == '__main__':
    _self_check()
