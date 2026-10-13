from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    fractional_order: float = 0.45
    fractional_window: int = 48
    normalization_window: int = 60
    pressure_window: int = 12
    pressure_threshold: float = 1.2
    volatility_window: int = 20
    target_volatility: float = 0.12
    atr_window: int = 14
    trailing_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1790882617'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        windows = (params.fractional_window, params.normalization_window,
                   params.pressure_window, params.volatility_window,
                   params.atr_window, params.holding_bars)
        if any(not isinstance(w, int) or w < 2 for w in windows):
            raise ValueError('Lookbacks and holding_bars must be integers >= 2')
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if any(not np.isfinite(v) or v <= 0 for v in
               (params.pressure_threshold, params.target_volatility,
                params.trailing_atr)):
            raise ValueError('Threshold, volatility target and ATR multiple must be positive')
        return int(max(params.fractional_window + params.normalization_window
                       + params.pressure_window + 1,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data['close'].astype(float).where(data['close'] > 0)
        close = close.replace([np.inf, -np.inf], np.nan)
        weights = np.ones(params.fractional_window, dtype=float)
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional = close.copy()
        for lag in range(1, params.fractional_window):
            fractional = fractional + weights[lag] * close.shift(lag)
        out = pd.DataFrame(index=data.index)
        out['fractional_close'] = fractional
        out['fractional_mean'] = fractional.rolling(params.normalization_window).mean()
        out['fractional_std'] = fractional.rolling(params.normalization_window).std(ddof=0)
        out['fractional_z'] = ((fractional - out['fractional_mean'])
                               / out['fractional_std'].replace(0.0, np.nan))
        out['overflow'] = (np.sign(out['fractional_z'])
                           * (out['fractional_z'].abs() - 0.5).clip(lower=0.0, upper=2.0))
        out['pressure'] = (out['overflow'].rolling(params.pressure_window).sum()
                           / np.sqrt(params.pressure_window))
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volatility'] = (out['return'].rolling(params.volatility_window).std(ddof=0)
                             * np.sqrt(252.0))
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['exposure'] = (params.target_volatility
                            / out['volatility'].replace(0.0, np.nan)).clip(0.05, 0.95)
        valid = (np.isfinite(close) & np.isfinite(out['pressure'])
                 & np.isfinite(out['atr']) & out['atr'].gt(0)
                 & np.isfinite(out['volatility']) & out['volatility'].gt(0))
        out['candidate'] = np.where(
            valid & out['pressure'].gt(params.pressure_threshold), 1,
            np.where(valid & out['pressure'].lt(-params.pressure_threshold), -1, 0),
        ).astype(int)
        out['confirmed'] = out['candidate'].where(
            out['candidate'].eq(out['candidate'].shift(1)), 0,
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        prices = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].to_numpy(dtype=int)
        volatility = indicators['volatility'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = 0
        entered = -1
        extreme = 0.0
        stop = np.nan
        for i in range(len(data)):
            price = prices[i]
            valid = (np.isfinite(price) and price > 0 and np.isfinite(atr[i])
                     and atr[i] > 0 and np.isfinite(volatility[i]) and volatility[i] > 0)
            if not valid:
                active = 0
            elif active:
                # Entry fills next bar; initialize from the first in-trade close.
                if i == entered + 1:
                    extreme = price
                    stop = price - active * params.trailing_atr * atr[i]
                if active == 1:
                    extreme = max(extreme, price)
                    stop = max(stop, extreme - params.trailing_atr * atr[i])
                    hit = price <= stop
                else:
                    # Mirror the high-water trail with a low-water trail for shorts.
                    extreme = min(extreme, price)
                    stop = min(stop, extreme + params.trailing_atr * atr[i])
                    hit = price >= stop
                if hit or i - entered >= params.holding_bars:
                    active = 0
            elif confirmed[i] != 0:
                active = int(confirmed[i])
                entered = i
            signals[i] = active
        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = indicators['exposure'].shift(1).replace(
            [np.inf, -np.inf], np.nan,
        ).fillna(0.05).clip(0.05, 0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=6)
    context = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    for side in (1, -1):
        prices = 100.0 + side * np.array([0, 0, 2, 4, -10, -10], dtype=float)
        data = pd.DataFrame({'open': prices, 'high': prices + 1,
                             'low': prices - 1, 'close': prices, 'volume': 1000.0}, index=index)
        derived = pd.DataFrame({'atr': 1.0, 'volatility': 0.2,
                                'confirmed': [0, side, side, side, side, side],
                                'exposure': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, index=index)
        result = strategy.generate_signals(data, derived, context, params).data
        assert result['signal'].tolist() == [0, 0, side, side, side, 0]
        assert np.isclose(result['size'].iloc[2], 0.2)
        assert result['size'].gt(0).all()
