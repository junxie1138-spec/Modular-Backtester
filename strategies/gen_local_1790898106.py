from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 10
    area_threshold: float = 0.08
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1790898106'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params):
        return int(max(200, params.signature_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        if params.signature_window < 2 or params.atr_window < 1 or params.hold_bars < 1:
            raise ValueError('Window lengths and holding period must be positive.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('stop_atr must be positive and finite.')
        if not np.isfinite(params.area_threshold) or not 0 < params.area_threshold < 0.5:
            raise ValueError('area_threshold must lie between zero and 0.5.')
        n = params.signature_window
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = np.log(close).diff()
        out['relative_range'] = (data['high'] - data['low']) / close
        out['range_change'] = out['relative_range'].diff()
        x, y = out['return'], out['range_change']
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, n):
            cross = x.shift(lag) * y - y.shift(lag) * x
            area = area + cross.rolling(n - lag, min_periods=n - lag).sum()
        out['area_raw'] = 0.5 * area
        out['area_scale'] = x.abs().rolling(n).sum() * y.abs().rolling(n).sum()
        out['area'] = out['area_raw'] / out['area_scale'].where(out['area_scale'] > 0)
        out['net_return'] = x.rolling(n).sum()
        out['deformation'] = y.rolling(n).sum()
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        area = indicators['area']
        move = indicators['net_return']
        deformation = indicators['deformation']
        expanding = deformation > 0
        contracting = deformation < 0
        upward_order = (move > 0) & (area < -params.area_threshold)
        downward_order = (move < 0) & (area > params.area_threshold)
        long_setup = (expanding & upward_order) | (contracting & downward_order)
        short_setup = (expanding & downward_order) | (contracting & upward_order)
        ready = np.isfinite(indicators[['area', 'net_return', 'deformation', 'ma200', 'atr']]).all(axis=1)
        ready &= np.isfinite(data['close']) & (data['close'] > 0) & (indicators['atr'] > 0)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[ready & long_setup & (data['close'] > indicators['ma200'])] = 1
        candidate.loc[ready & short_setup & (data['close'] < indicators['ma200'])] = -1
        wanted = candidate.to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop = np.nan
        distance = 0.0
        age = 0
        for i in range(len(data)):
            if position:
                # A decision from the preceding close fills at this bar's open.
                if np.isnan(entry):
                    if not np.isfinite(opening[i]) or opening[i] <= 0:
                        position = 0
                        continue
                    entry = opening[i]
                    stop = entry - position * distance
                age += 1
                stopped = np.isfinite(close[i]) and position * (close[i] - stop) < 0
                if stopped or age >= params.hold_bars:
                    position = 0
                    entry = np.nan
                    continue
            elif wanted[i]:
                position = int(wanted[i])
                entry = np.nan
                distance = params.stop_atr * atr[i]
                age = 0
            decisions[i] = position
        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': [100.0] * 8,
        'high': [125.0] * 8,
        'low': [90.0] * 8,
        'close': [120.0, 99.0, 99.0, 97.0, 100.0, 101.0, 103.0, 100.0],
        'volume': [1000.0] * 8,
    }, index=index)
    indicators = pd.DataFrame({
        'area': [-0.2] * 4 + [0.2] * 4,
        'net_return': [0.1] * 4 + [-0.1] * 4,
        'deformation': [0.01] * 8,
        'ma200': [90.0] * 4 + [110.0] * 4,
        'atr': [1.0, 50.0, 50.0, 50.0, 1.0, 50.0, 50.0, 50.0],
    }, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams(stop_atr=2.0)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert (result['size'] > 0).all()
    expiry = strategy.generate_signals(data, indicators, ctx, SignatureParams(stop_atr=100.0, hold_bars=2)).data
    assert expiry['signal'].iloc[:4].tolist() == [0, 1, 1, 0]
