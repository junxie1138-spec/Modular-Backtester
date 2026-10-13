from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_bars: int = 12
    edge_fraction: float = 0.65
    atr_bars: int = 14
    stop_atr: float = 0.8
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local-w2_1791210794'

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        if params.graph_bars < 3 or params.atr_bars < 1:
            raise ValueError('Lookbacks must be positive; graph_bars must be at least 3.')
        if not 0.0 < params.edge_fraction <= 1.0:
            raise ValueError('edge_fraction must be in (0, 1].')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError('stop_atr must be finite and positive.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be 1 or 2.')
        return int(max(200, params.graph_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        result = pd.DataFrame(index=data.index)
        result['ma200'] = close.rolling(200, min_periods=200).mean()
        previous = close.shift(1)
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        tr = tr.where(data[['high', 'low', 'close']].notna().all(axis=1) & previous.notna())
        result['atr'] = tr.replace([np.inf, -np.inf], np.nan).rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = pd.Series(0.0, index=data.index)
        minimum = pd.Series(np.inf, index=data.index)
        maximum = pd.Series(-np.inf, index=data.index)
        # An edge exists when every intervening close lies strictly below
        # its connecting line; reversing the inequality gives the inverted graph.
        for lag in range(1, params.graph_bars + 1):
            slope = (close - close.shift(lag)) / lag
            upper_degree += (slope < minimum).astype(float)
            lower_degree += (slope > maximum).astype(float)
            minimum = np.minimum(minimum, slope)
            maximum = np.maximum(maximum, slope)
        complete = close.rolling(params.graph_bars + 1).count().eq(params.graph_bars + 1)
        result['upper_degree'] = upper_degree.where(complete)
        result['lower_degree'] = lower_degree.where(complete)
        result['displacement'] = close.diff(2)
        threshold = params.graph_bars * params.edge_fraction
        ready = result.notna().all(axis=1) & result['atr'].gt(0.0)
        long_entry = (
            ready & close.gt(result['ma200']) & result['displacement'].gt(0.0)
            & result['upper_degree'].ge(threshold)
            & result['upper_degree'].gt(result['lower_degree'])
        )
        short_entry = (
            ready & close.lt(result['ma200']) & result['displacement'].lt(0.0)
            & result['lower_degree'].ge(threshold)
            & result['lower_degree'].gt(result['upper_degree'])
        )
        result['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return result

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        regimes = indicators['ma200'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        high_stop = -np.inf
        short_stop = np.inf
        for i in range(len(data)):
            close, atr, regime = closes[i], atrs[i], regimes[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0.0 and np.isfinite(regime)
            if position:
                age += 1
                if not valid:
                    position = 0
                else:
                    # Start tracking closes on the actual fill bar, not the
                    # previous close on which the entry decision was made.
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    low_water = close if np.isnan(low_water) else min(low_water, close)
                    high_stop = max(high_stop, high_water - params.stop_atr * atr)
                    short_stop = min(short_stop, low_water + params.stop_atr * atr)
                    # The required high-water exit applies to both directions.
                    # Shorts also have a mirrored protective trailing exit.
                    exit_trade = close < high_stop or age >= params.hold_bars
                    exit_trade |= position == -1 and close > short_stop
                    exit_trade |= position == 1 and close <= regime
                    exit_trade |= position == -1 and close >= regime
                    if exit_trade:
                        position = 0
                signals[i] = position
                # Preserve an explicit flat bar after an exit.
                continue
            if valid and entries[i] in (-1, 1):
                position = int(entries[i])
                age = 0
                high_water = low_water = np.nan
                high_stop = -np.inf
                short_stop = np.inf
                signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range('2020-01-01', periods=240)
    close = pd.Series(100.0 + np.arange(240, dtype=float) ** 2 / 1000.0, index=index)
    data = pd.DataFrame({
        'open': close, 'high': close + 1.0, 'low': close - 1.0,
        'close': close, 'volume': 1000.0,
    })
    topology = strategy.indicators(data, params)
    assert topology['upper_degree'].iloc[-1] == params.graph_bars
    assert topology['lower_degree'].iloc[-1] == 1.0
    assert topology['entry'].iloc[:199].eq(0).all()
    fixture = pd.DataFrame({
        'atr': [1.0, 1.0, 5.0, 1.0],
        'ma200': [90.0] * 4,
        'entry': [1, 0, 0, 0],
    }, index=index[:4])
    sample = data.iloc[:4].copy()
    sample['close'] = [100.0, 110.0, 108.0, 108.0]
    frame = strategy.generate_signals(sample, fixture, None, params).data
    assert frame['signal'].tolist() == [0, 1, 1, 0]
    assert frame['size'].gt(0.0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
