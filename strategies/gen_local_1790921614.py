from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_window: int = 18
    forecast_window: int = 5
    innovation_window: int = 42
    range_window: int = 63
    atr_window: int = 14
    innovation_threshold: float = 0.8
    edge_buffer: float = 0.15
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1790921614'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(
            params.range_window + 1,
            params.atr_window + 1,
            params.trend_window + params.forecast_window + params.innovation_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.trend_window, params.forecast_window, params.innovation_window, params.range_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars')
        if not (0 < params.edge_buffer < 0.5) or params.innovation_threshold <= 0 or params.stop_atr <= 0 or params.holding_bars < 1:
            raise ValueError('Invalid entry or exit parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['log_close'] = np.log(close)
        out['time'] = np.arange(len(data), dtype=float)
        variance = (params.trend_window ** 2 - 1) / 12.0
        out['latent_slope'] = out['log_close'].rolling(params.trend_window).cov(out['time'], ddof=0) / variance
        out['predicted_slope'] = out['latent_slope'].shift(1).rolling(params.forecast_window).mean()
        out['innovation'] = out['latent_slope'] - out['predicted_slope']
        out['innovation_mean'] = out['innovation'].shift(1).rolling(params.innovation_window).mean()
        out['innovation_scale'] = out['innovation'].shift(1).rolling(params.innovation_window).std(ddof=0)
        scale = out['innovation_scale'].where(out['innovation_scale'] > 1e-12)
        out['innovation_z'] = (out['innovation'] - out['innovation_mean']) / scale
        out['range_low'] = data['low'].shift(1).rolling(params.range_window).min()
        out['range_high'] = data['high'].shift(1).rolling(params.range_window).max()
        out['range_width'] = out['range_high'] - out['range_low']
        out['relative_position'] = (close - out['range_low']) / out['range_width'].where(out['range_width'] > 0)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        ready = np.isfinite(out[['latent_slope', 'innovation_z', 'relative_position', 'atr']]).all(axis=1) & out['atr'].gt(0)
        out['long_setup'] = ready & out['latent_slope'].gt(0) & out['innovation_z'].ge(params.innovation_threshold) & out['relative_position'].between(0.5, 1.0 - params.edge_buffer)
        out['short_setup'] = ready & out['latent_slope'].lt(0) & out['innovation_z'].le(-params.innovation_threshold) & out['relative_position'].between(params.edge_buffer, 0.5)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        long_setup = indicators['long_setup'].fillna(False).to_numpy(dtype=bool)
        short_setup = indicators['short_setup'].fillna(False).to_numpy(dtype=bool)
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # The preceding decision fills at this bar's open.
                    if not np.isfinite(opening[i]) or opening[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        position = 0
                        raw[i] = 0
                        continue
                    stop = opening[i] - position * params.stop_atr * atr[i - 1]
                stopped = np.isfinite(close[i]) and position * (close[i] - stop) < 0
                expired = i - entry_bar >= params.holding_bars - 1
                if stopped or expired:
                    position = 0
                raw[i] = position
                continue
            if i > 0 and np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                if long_setup[i] and long_setup[i - 1]:
                    position = 1
                elif short_setup[i] and short_setup[i - 1]:
                    position = -1
                if position:
                    entry_bar = i + 1
            raw[i] = position
        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({'open': 100.0, 'high': 104.0, 'low': 95.0, 'close': [100, 100, 100, 101, 96, 100, 100, 103, 100], 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'long_setup': [True, True, False, False, False, False, False, False, False], 'short_setup': [False, False, False, False, False, True, True, False, False]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, 0, -1, 0]
    assert result['size'].gt(0).all()
    result = GeneratedStrategy().generate_signals(data, indicators, None, StrategyParams(holding_bars=2)).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0, 0, -1, 0]
