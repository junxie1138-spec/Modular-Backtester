from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    forecast_threshold: float = 0.55
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1791200560'

    @classmethod
    def params_type(cls):
        return AnalogParams

    @staticmethod
    def warmup_bars(params: AnalogParams) -> int:
        return 160

    @staticmethod
    def indicators(data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not np.isfinite(params.forecast_threshold) or params.forecast_threshold <= 0:
            raise ValueError('forecast_threshold must be finite and positive')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be finite and positive')
        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        out['atr_fraction'] = out['atr'] / data['close'].where(data['close'] > 0)
        out['vol_base'] = out['atr_fraction'].rolling(60, min_periods=60).mean()
        out['vol_block'] = out['atr_fraction'].rolling(5, min_periods=5).mean()
        base = out['vol_base'].where(out['vol_base'] > 0)
        for lag in (0, 5, 10):
            ratio = out['vol_block'].shift(lag) / base
            out[f'shape_{lag}'] = np.log(ratio.where(ratio > 0))
        out['return'] = data['close'].pct_change(fill_method=None)
        out['return_std'] = out['return'].rolling(5, min_periods=5).std(ddof=0)
        dispersion = out['return_std'] / out['vol_block'].where(out['vol_block'] > 0)
        out['dispersion'] = np.log(dispersion.where(dispersion > 0))
        # At t, this is the completed return from the open at t-7 through t's close.
        out['completed_return'] = data['close'] / data['open'].shift(7).where(data['open'].shift(7) > 0) - 1.0
        out['completed_outcome'] = out['completed_return'] / out['atr_fraction'].shift(8).where(out['atr_fraction'].shift(8) > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        features = out[['shape_0', 'shape_5', 'shape_10', 'dispersion']].to_numpy(dtype=float)
        outcomes = out['completed_outcome'].to_numpy(dtype=float)
        forecast = np.full(len(data), np.nan)
        agreement = np.full(len(data), np.nan)
        distance = np.full(len(data), np.nan)
        # ponytail: bounded 504-candidate scan; use an indexed neighbor search only if histories become much larger.
        for i in range(159, len(data)):
            if not np.isfinite(features[i]).all():
                continue
            # An analog ends at j, its outcome ends at j+8, and the current shape starts at i-14.
            candidates = np.arange(max(0, i - 526), i - 22)
            valid = np.isfinite(features[candidates]).all(axis=1) & np.isfinite(outcomes[candidates + 8])
            candidates = candidates[valid]
            if len(candidates) < 60:
                continue
            distances = np.sqrt(np.mean((features[candidates] - features[i]) ** 2, axis=1))
            nearest = np.argsort(distances, kind='stable')[:12]
            labels = outcomes[candidates[nearest] + 8]
            forecast[i] = np.median(labels)
            agreement[i] = np.mean(np.sign(labels) == np.sign(forecast[i]))
            distance[i] = np.median(distances[nearest])
        out['forecast'] = forecast
        out['agreement'] = agreement
        out['analog_distance'] = distance
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AnalogParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        forecast = indicators['forecast'].to_numpy(dtype=float)
        agreement = indicators['agreement'].to_numpy(dtype=float)
        distance = indicators['analog_distance'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        entry = np.nan
        floor = -np.inf
        armed = False
        age = 0
        for i in range(len(data)):
            if side != 0:
                # The prior close's decision fills at this bar's open.
                if np.isnan(entry):
                    if i == 0 or not np.isfinite(opening[i]) or opening[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        side = 0
                        continue
                    entry = opening[i]
                    floor = -params.trail_atr * atr[i - 1]
                    age = 0
                    armed = False
                age += 1
                if not np.isfinite(close[i]) or close[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    entry = np.nan
                    continue
                profit = side * (close[i] - entry)
                if profit >= 0.01 * entry:
                    armed = True
                if armed:
                    # This profit-coordinate stop only increases for either direction.
                    # Long price stops rise; short price stops fall toward favorable prices.
                    floor = max(floor, 0.0, profit - params.trail_atr * atr[i])
                # Stops are evaluated at the close and exits fill at the next open.
                if profit <= floor or age >= 8:
                    side = 0
                    entry = np.nan
                raw[i] = side
                continue
            eligible = (
                np.isfinite(close[i]) and close[i] > 0
                and np.isfinite(atr[i]) and atr[i] > 0
                and np.isfinite(forecast[i])
                and np.isfinite(agreement[i]) and agreement[i] >= 2.0 / 3.0
                and np.isfinite(distance[i]) and distance[i] <= 0.25
            )
            if eligible and abs(forecast[i]) >= params.forecast_threshold:
                side = 1 if forecast[i] > 0 else -1
                entry = np.nan
                raw[i] = side
        df = pd.DataFrame({'signal': raw, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = AnalogParams()
    index = pd.date_range('2020-01-01', periods=12, freq='D')
    for direction in (1, -1):
        closes = 100.0 + direction * np.array([0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        data = pd.DataFrame({'open': 100.0, 'high': np.maximum(100.0, closes) + 1.0, 'low': np.minimum(100.0, closes) - 1.0, 'close': closes, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'atr': 1.0, 'forecast': float(direction), 'agreement': 1.0, 'analog_distance': 0.0}, index=index)
        result = GeneratedStrategy.generate_signals(data, ind, None, params).data
        assert result['signal'].iloc[:6].tolist() == [0, direction, direction, direction, direction, 0]
        assert (result['size'] > 0).all()
        ind['forecast'] = np.nan
        assert (GeneratedStrategy.generate_signals(data, ind, None, params).data['signal'] == 0).all()
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 320)))
    opening = np.concatenate(([100.0], close[:-1]))
    data = pd.DataFrame({'open': opening, 'high': np.maximum(opening, close) * 1.01, 'low': np.minimum(opening, close) * 0.99, 'close': close, 'volume': 1000.0}, index=pd.date_range('2020-01-01', periods=320, freq='D'))
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:280], params)
    pd.testing.assert_frame_equal(full.iloc[:280], prefix)
    full_signals = GeneratedStrategy.generate_signals(data, full, None, params).data
    prefix_signals = GeneratedStrategy.generate_signals(data.iloc[:280], prefix, None, params).data
    pd.testing.assert_frame_equal(full_signals.iloc[:280], prefix_signals)


if __name__ == '__main__':
    _self_check()
