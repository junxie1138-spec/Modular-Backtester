from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    shape_bars: int = 12
    history_bars: int = 504
    neighbors: int = 5
    hold_bars: int = 18
    min_forecast: float = 0.01
    max_distance: float = 0.35

    def __post_init__(self):
        if self.shape_bars < 3 or self.neighbors < 1 or self.hold_bars < 1:
            raise ValueError('Shape, neighbor count, and holding period must be positive.')
        if self.history_bars < (self.neighbors + 1) * self.hold_bars:
            raise ValueError('Historical window is too short for the requested analogs.')
        if not np.isfinite(self.min_forecast):
            raise ValueError('Forecast threshold must be finite.')
        if not np.isfinite(self.max_distance) or self.max_distance <= 0:
            raise ValueError('Maximum distance must be finite and positive.')


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local_1791285460'

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(200, 63 + params.shape_bars + params.history_bars + params.hold_bars + 2))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['fast_vol'] = out['return'].rolling(5, min_periods=5).std(ddof=0)
        out['base_vol'] = out['return'].rolling(63, min_periods=63).std(ddof=0)
        ratio = out['fast_vol'].div(out['base_vol'].where(out['base_vol'] > 0))
        out['log_vol_ratio'] = np.log(ratio.where(ratio > 0)).replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['regime'] = close.gt(out['ma200']).fillna(False)
        opening = data['open'].where(data['open'] > 0)
        out['matured_return'] = opening.div(opening.shift(params.hold_bars)).sub(1).replace([np.inf, -np.inf], np.nan)
        columns = []
        for k in range(4):
            name = f'vol_shape_{k}'
            out[name] = out['log_vol_ratio'].shift(k * params.shape_bars // 3)
            columns.append(name)
        shapes = out[columns].to_numpy(dtype=float)
        regime = out['regime'].to_numpy(dtype=bool)
        outcomes = out['matured_return'].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        win_fraction = np.full(len(data), np.nan)
        distance_used = np.full(len(data), np.nan)
        for i in range(self.warmup_bars(params) - 1, len(data)):
            if not regime[i] or not np.isfinite(shapes[i]).all():
                continue
            # Spacing analog endpoints by the holding period avoids overlapping outcome windows.
            candidates = np.arange(max(0, i - params.history_bars), i - params.hold_bars, params.hold_bars, dtype=int)
            endpoints = candidates + params.hold_bars + 1
            valid = regime[candidates] & np.isfinite(shapes[candidates]).all(axis=1) & np.isfinite(outcomes[endpoints])
            candidates = candidates[valid]
            if len(candidates) < params.neighbors:
                continue
            distances = np.sqrt(np.mean((shapes[candidates] - shapes[i]) ** 2, axis=1))
            nearest = np.argsort(distances)[:params.neighbors]
            if distances[nearest[-1]] > params.max_distance:
                continue
            # Endpoint j forecasts entry at open j+1 and exit at open j+hold_bars+1.
            matched = outcomes[candidates[nearest] + params.hold_bars + 1]
            forecast[i] = matched.mean()
            win_fraction[i] = np.mean(matched > 0)
            distance_used[i] = distances[nearest].mean()
        out['forecast'] = forecast
        out['win_fraction'] = win_fraction
        out['analog_distance'] = distance_used
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        entry = (indicators['regime'].fillna(False) & indicators['forecast'].gt(params.min_forecast) & indicators['win_fraction'].ge(0.6)).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entered = -1
        for i in range(len(data)):
            if entered >= 0:
                if i - entered >= params.hold_bars:
                    entered = -1
                    continue
                signal[i] = 1
            elif entry[i]:
                entered = i
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = AnalogParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2000-01-01', periods=2 * params.hold_bars + 5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'regime': True, 'forecast': 0.05, 'win_fraction': 1.0}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = [0] + [1] * params.hold_bars + [0] + [1] * params.hold_bars + [0]
    assert result['signal'].iloc[:len(expected)].tolist() == expected
    assert result['size'].gt(0).all()
    n = strategy.warmup_bars(params) + 45
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.001, 0.008, n)))
    bars = pd.DataFrame({'open': prices, 'high': prices * 1.01, 'low': prices * 0.99, 'close': prices, 'volume': 1000000.0}, index=pd.date_range('2000-01-01', periods=n))
    full = strategy.indicators(bars, params)
    prefix = strategy.indicators(bars.iloc[:-20], params)
    pd.testing.assert_frame_equal(full.iloc[:-20], prefix)


if __name__ == '__main__':
    _self_check()
