from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilitySeasonParams:
    visibility_window: int = 12
    season_window: int = 252
    degree_margin: float = 0.15
    seasonal_min: float = 0.001
    min_samples: int = 25
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilitySeasonParams]):
    strategy_id = 'gen_local_1791340667'

    @classmethod
    def params_type(cls):
        return VisibilitySeasonParams

    def warmup_bars(self, params):
        return int(max(params.visibility_window, params.atr_window + 1,
                       params.season_window + params.hold_bars + 1))

    def indicators(self, data, params):
        if (params.visibility_window < 3 or params.season_window < 5
                or params.atr_window < 1 or params.hold_bars < 1
                or params.min_samples < 1 or params.stop_atr <= 0
                or not np.isfinite(params.stop_atr)
                or not np.isfinite(params.degree_margin)
                or not np.isfinite(params.seasonal_min)):
            raise ValueError('Invalid strategy parameters')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0)
        high = data['high'].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data['low'].astype(float).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - previous).abs(),
                        (low - previous).abs()], axis=1).max(axis=1, skipna=False)
        out['atr'] = tr.rolling(params.atr_window).mean()

        prices = close.to_numpy()
        peak_degree = np.zeros(len(data), dtype=float)
        trough_degree = np.zeros(len(data), dtype=float)
        maximum_slope = np.full(len(data), -np.inf)
        minimum_slope = np.full(len(data), np.inf)
        # Endpoint visibility is a strict record in backward chord slopes.
        for distance in range(1, params.visibility_window):
            slope = (close.shift(distance).to_numpy() - prices) / distance
            valid = np.isfinite(slope)
            peak_degree += valid & (slope > maximum_slope)
            trough_degree += valid & (slope < minimum_slope)
            maximum_slope = np.where(valid, np.maximum(maximum_slope, slope), maximum_slope)
            minimum_slope = np.where(valid, np.minimum(minimum_slope, slope), minimum_slope)

        graph_ready = close.rolling(params.visibility_window).count().eq(params.visibility_window)
        out['peak_degree'] = pd.Series(peak_degree, index=data.index).where(graph_ready)
        out['trough_degree'] = pd.Series(trough_degree, index=data.index).where(graph_ready)
        out['degree_imbalance'] = (out['peak_degree'] - out['trough_degree']) / (params.visibility_window - 1)

        # Historical outcomes are available only after completion, plus one bar.
        out['matured_return'] = (close / close.shift(params.hold_bars) - 1.0).shift(1)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out['origin_weekday'] = weekday.shift(params.hold_bars + 1)
        out['baseline_return'] = out['matured_return'].rolling(params.season_window).mean()
        out['seasonal_excess'] = np.nan
        out['seasonal_samples'] = np.nan
        for day in range(5):
            belongs = out['origin_weekday'].eq(day) & out['matured_return'].notna()
            count = belongs.astype(float).rolling(params.season_window).sum()
            total = out['matured_return'].where(belongs, 0.0).rolling(params.season_window).sum()
            mean = total / count.replace(0.0, np.nan)
            excess = (mean - out['baseline_return']).where(count >= params.min_samples)
            current = weekday.eq(day)
            out.loc[current, 'seasonal_excess'] = excess.loc[current]
            out.loc[current, 'seasonal_samples'] = count.loc[current]

        out['eligible'] = (out['degree_imbalance'].ge(params.degree_margin)
                           & out['seasonal_excess'].ge(params.seasonal_min)
                           & out['atr'].gt(0) & close.notna()).fillna(False)
        out['confirmed'] = out['eligible'] & out['eligible'].shift(1, fill_value=False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        prices = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        confirmed = indicators['confirmed'].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        highest = stop = -np.inf
        age = 0

        for i, price in enumerate(prices):
            if not np.isfinite(price) or price <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                active = False
                continue
            if active:
                age += 1
                # Entry fills on the next bar; start its close high-water mark there.
                highest = max(highest, price)
                stop = max(stop, highest - params.stop_atr * atr[i])
                if price <= stop or age >= params.hold_bars:
                    active = False
                    continue
                raw[i] = 1
            elif confirmed[i]:
                active = True
                highest = stop = -np.inf
                age = 0
                raw[i] = 1

        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({'close': [100., 100., 103., 102., 98., 99.]}, index=index)
    indicators = pd.DataFrame({'atr': [1., 1., 1., 3., 3., 1.],
                               'confirmed': [False, True, True, True, True, True]}, index=index)
    result = strategy.generate_signals(data, indicators, None, VisibilitySeasonParams(stop_atr=2.0))
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 1, 0]
    result = strategy.generate_signals(data, indicators, None,
                                      VisibilitySeasonParams(stop_atr=100.0, hold_bars=2))
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 0, 1]
    assert result.data['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
