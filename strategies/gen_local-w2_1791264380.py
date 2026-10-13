from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    analog_window: int = 126
    match_radius: float = 0.75
    min_edge: float = 0.001
    overload_ratio: float = 1.25
    relief_ratio: float = 0.85
    trailing_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791264380'

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        # Maximum lag is window + 4, plus two shape lags and 21 volatility bars.
        return int(params.analog_window + 27)

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        values = [params.match_radius, params.min_edge, params.overload_ratio,
                  params.relief_ratio, params.trailing_atr]
        if (params.analog_window < 5 or not np.isfinite(values).all()
                or params.match_radius <= 0 or params.min_edge < 0
                or params.overload_ratio <= 1
                or not 0 < params.relief_ratio < 1 or params.trailing_atr <= 0):
            raise ValueError('Invalid analog strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None)
        out['volatility'] = out['return'].rolling(20, min_periods=20).std()
        out['normalized_return'] = out['return'].div(
            out['volatility'].where(out['volatility'] > 0)
        )
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['volume_baseline'] = data['volume'].rolling(20, min_periods=20).median()
        out['volume_ratio'] = data['volume'].div(
            out['volume_baseline'].where(out['volume_baseline'] > 0)
        )
        out['prior_volume_ratio'] = out['volume_ratio'].shift(1)
        out['volume_relief'] = (
            (out['prior_volume_ratio'] >= params.overload_ratio)
            & (out['volume_ratio'] <= params.relief_ratio * out['prior_volume_ratio'])
            & (out['volume_ratio'] >= 0.5)
        )
        z = out['normalized_return']
        total_weight = pd.Series(0.0, index=data.index)
        total_outcome = pd.Series(0.0, index=data.index)
        matches = pd.Series(0, index=data.index, dtype=int)
        # Each historical outcome ends before the current three-bar shape begins.
        for lag in range(5, params.analog_window + 5):
            distance_squared = sum(
                (z.shift(j) - z.shift(lag + j)).pow(2) for j in range(3)
            ) / 3.0
            outcome = (1.0 + out['return'].shift(lag - 1)) * (
                1.0 + out['return'].shift(lag - 2)
            ) - 1.0
            valid = (
                np.isfinite(distance_squared) & np.isfinite(outcome)
                & (distance_squared <= params.match_radius ** 2)
            )
            weight = np.exp(-distance_squared / (2.0 * params.match_radius ** 2))
            weight = weight.where(valid, 0.0).fillna(0.0)
            total_weight += weight
            total_outcome += weight * outcome.where(valid, 0.0).fillna(0.0)
            matches += valid.astype(int)
        out['analog_weight'] = total_weight
        out['analog_matches'] = matches
        out['analog_edge'] = total_outcome.div(total_weight.where(total_weight > 0))
        out['analog_agrees'] = (matches >= 5) & (out['analog_edge'] >= params.min_edge)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        entry = (indicators['analog_agrees'].fillna(False)
                 & indicators['volume_relief'].fillna(False)).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = np.nan
        entry_atr = np.nan
        first_ready = self.warmup_bars(params) - 1
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if not valid:
                    active = False
                    continue
                if age == 1:
                    # Entry fills at this bar's open; only held-bar closes enter the watermark.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    high_water = closes[i]
                    stop = opens[i] - params.trailing_atr * entry_atr
                else:
                    high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trailing_atr * atr[i])
                if closes[i] <= stop or age >= 2:
                    active = False
                else:
                    signal[i] = 1
            elif i >= first_ready and valid and entry[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = np.nan
                entry_atr = atr[i]
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = AnalogParams(analog_window=5)
    strategy = GeneratedStrategy()
    n = strategy.warmup_bars(params) + 4
    index = pd.date_range('2020-01-01', periods=n, freq='D')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'analog_agrees': False, 'volume_relief': False,
                               'atr': 1.0}, index=index)
    t = strategy.warmup_bars(params) - 1
    indicators.iloc[t, indicators.columns.get_loc('analog_agrees')] = True
    indicators.iloc[t, indicators.columns.get_loc('volume_relief')] = True
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[t:t + 4].tolist() == [0, 1, 1, 0]
    data.iloc[t + 1, data.columns.get_loc('close')] = 98.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].iloc[t:t + 4].tolist() == [0, 1, 0, 0]
    indicators['volume_relief'] = False
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].eq(0).all()
    assert result['size'].gt(0).all()
