from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 5
    volatility_window: int = 6
    innovation_threshold: float = 0.15
    profit_target: float = 0.025
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791262840'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        if not 2 <= params.slope_window <= 7:
            raise ValueError('slope_window must be between 2 and 7')
        if not 2 <= params.volatility_window <= 9:
            raise ValueError('volatility_window must be between 2 and 9')
        return int(max(params.slope_window + 3, params.volatility_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        result = pd.DataFrame(index=data.index)
        log_close = np.log(data['close'].where(data['close'] > 0))
        returns = log_close.diff()
        window = params.slope_window
        center = (window - 1) / 2.0
        denominator = window * (window * window - 1) / 12.0
        slope = sum((k - center) * log_close.shift(window - 1 - k)
                    for k in range(window)) / denominator
        forecast = slope.shift(1).rolling(3, min_periods=3).mean()
        volatility = returns.rolling(params.volatility_window,
                                     min_periods=params.volatility_window).std(ddof=0)
        innovation = ((slope - forecast) / volatility.where(volatility > 0)).replace(
            [np.inf, -np.inf], np.nan)
        seasonal_gate = ((data.index.days_in_month - data.index.day < 7)
                         & (data.index.dayofweek >= 1)
                         & (data.index.dayofweek <= 3))
        result['log_close'] = log_close.fillna(0.0)
        result['return'] = returns.fillna(0.0)
        result['slope'] = slope.fillna(0.0)
        result['slope_forecast'] = forecast.fillna(0.0)
        result['volatility'] = volatility.fillna(0.0)
        result['innovation'] = innovation.fillna(0.0)
        result['seasonal_gate'] = seasonal_gate
        result['ready'] = innovation.notna() & slope.notna()
        return result

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        if not np.isfinite(params.innovation_threshold) or params.innovation_threshold <= 0:
            raise ValueError('innovation_threshold must be positive and finite')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('profit_target must be positive and finite')
        if params.max_hold_bars < 1:
            raise ValueError('max_hold_bars must be positive')
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        innovation = indicators['innovation'].to_numpy(dtype=float)
        slope = indicators['slope'].to_numpy(dtype=float)
        eligible = (indicators['ready'] & indicators['seasonal_gate']).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if not np.isfinite(entry_price):
                    entry_price = opens[i]
                    entry_bar = i
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                        entry_price = np.nan
                        continue
                gain = position * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_hit = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or time_hit:
                    position = 0
                    entry_price = np.nan
                    continue
            elif (eligible[i] and np.isfinite(closes[i]) and closes[i] > 0
                  and abs(innovation[i]) >= params.innovation_threshold
                  and innovation[i] * slope[i] > 0):
                position = 1 if innovation[i] > 0 else -1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.bdate_range('2024-01-01', periods=30)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000000.0}, index=index)
    params = Params(max_hold_bars=18)
    assert GeneratedStrategy.warmup_bars(params) <= 10
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index) and not derived.isna().any().any()
    for direction in (1, -1):
        injected = pd.DataFrame({'innovation': 0.0, 'slope': 0.0,
                                 'ready': True, 'seasonal_gate': False}, index=index)
        injected.iloc[0] = [float(direction), float(direction), True, True]
        timed = GeneratedStrategy.generate_signals(data, injected, None, params).data
        assert timed['signal'].iloc[0] == 0
        assert (timed['signal'].iloc[1:19] == direction).all()
        assert (timed['signal'].iloc[19:] == 0).all()
        profitable = data.copy()
        profitable.loc[index[4], 'close'] = 100.0 * (1.0 + direction * 0.03)
        targeted = GeneratedStrategy.generate_signals(profitable, injected, None, params).data
        assert (targeted['signal'].iloc[1:5] == direction).all()
        assert (targeted['signal'].iloc[5:] == 0).all()
        assert targeted['signal'].isin([-1, 0, 1]).all()
        assert (targeted['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
