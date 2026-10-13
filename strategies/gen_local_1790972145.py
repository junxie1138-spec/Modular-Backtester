from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    pressure_window: int = 3
    normalization_window: int = 40
    correlation_window: int = 80
    correlation_ceiling: float = -0.10
    extreme_threshold: float = 0.8
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = 'gen_local_1790972145'

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return int(max(params.fractional_window, params.pressure_window) + params.normalization_window + params.holding_bars + params.correlation_window)

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if min(params.fractional_window, params.normalization_window, params.correlation_window) < 2:
            raise ValueError('fractional, normalization and correlation windows must exceed one')
        if params.pressure_window < 1 or params.holding_bars < 1:
            raise ValueError('pressure_window and holding_bars must be positive')
        if not -1.0 < params.correlation_ceiling < 0.0 or params.extreme_threshold <= 0.0:
            raise ValueError('Require a negative correlation ceiling and positive extreme threshold')

        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1) / k
        fractional_close = close * weights[0]
        for k in range(1, params.fractional_window):
            fractional_close = fractional_close + weights[k] * close.shift(k)

        out = pd.DataFrame(index=data.index)
        out['fractional_close'] = fractional_close
        out['bar_range'] = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['body_pressure'] = ((data['close'] - data['open']) / out['bar_range']).clip(-1.0, 1.0)
        out['pressure'] = out['body_pressure'].rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        for name, source in [('fractional', 'fractional_close'), ('pressure', 'pressure')]:
            series = out[source]
            out[name + '_mean'] = series.rolling(params.normalization_window, min_periods=params.normalization_window).mean()
            out[name + '_std'] = series.rolling(params.normalization_window, min_periods=params.normalization_window).std(ddof=0)
            denominator = out[name + '_std'].where(out[name + '_std'] > 0.0)
            out[name + '_z'] = (series - out[name + '_mean']) / denominator
            out[name + '_lag'] = series.shift(params.holding_bars)
            out[name + '_corr'] = series.rolling(params.correlation_window, min_periods=params.correlation_window).corr(out[name + '_lag'])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveParams) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError('holding_bars must be positive')
        valid = np.isfinite(indicators[['fractional_z', 'pressure_z', 'fractional_corr', 'pressure_corr']]).all(axis=1)
        oscillating = valid & (indicators['fractional_corr'] < params.correlation_ceiling) & (indicators['pressure_corr'] < params.correlation_ceiling)
        long_entry = oscillating & (indicators['fractional_z'] < -params.extreme_threshold) & (indicators['pressure_z'] < -params.extreme_threshold)
        short_entry = oscillating & (indicators['fractional_z'] > params.extreme_threshold) & (indicators['pressure_z'] > params.extreme_threshold)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
                else:
                    signal[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = i + params.holding_bars
                signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = WaveParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=2 * params.holding_bars + 4, freq='B')
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({'fractional_z': -2.0, 'pressure_z': -2.0, 'fractional_corr': -0.5, 'pressure_corr': -0.5}, index=index)
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals['signal'].iloc[0] == 0
    assert (signals['signal'].iloc[1:params.holding_bars + 1] == 1).all()
    assert signals['signal'].iloc[params.holding_bars + 1] == 0
    assert signals['signal'].iloc[params.holding_bars + 2] == 1
    indicators['pressure_z'] = 2.0
    assert (strategy.generate_signals(data, indicators, None, params).data['signal'] == 0).all()
    indicators['fractional_z'] = 2.0
    assert (strategy.generate_signals(data, indicators, None, params).data['signal'].iloc[1:params.holding_bars + 1] == -1).all()
    indicators.loc[:, :] = np.nan
    assert (strategy.generate_signals(data, indicators, None, params).data['signal'] == 0).all()


if __name__ == '__main__':
    _self_check()
