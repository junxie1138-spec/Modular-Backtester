from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_window: int = 10
    area_threshold: float = 0.04
    pressure_window: int = 8
    pressure_threshold: float = 0.12
    atr_window: int = 14
    atr_multiple: float = 2.5
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1791326970'

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.signature_window + 1, params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if min(params.signature_window, params.pressure_window, params.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two.')
        if params.max_hold_bars < 1 or params.atr_multiple <= 0:
            raise ValueError('Holding period and ATR multiple must be positive.')
        if not np.isfinite(params.atr_multiple) or not np.isfinite(params.area_threshold) or not np.isfinite(params.pressure_threshold):
            raise ValueError('Thresholds must be finite.')
        close = data['close'].astype(float).where(data['close'] > 0)
        opening = data['open'].astype(float).where(data['open'] > 0)
        overnight = np.log(opening / close.shift(1))
        intraday = np.log(close / opening)
        length = params.signature_window
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature of the gap/intraday path.
        for lag in range(1, length):
            cross = overnight.shift(lag) * intraday - intraday.shift(lag) * overnight
            area = area + 0.5 * cross.rolling(length - lag, min_periods=length - lag).sum()
        scale = overnight.abs().rolling(length).sum() * intraday.abs().rolling(length).sum()
        normalized_area = area / scale.where(scale > 0)
        displacement = (overnight + intraday).rolling(length).sum()
        span = (data['high'] - data['low']).astype(float)
        location = ((2.0 * close - data['high'] - data['low']) / span.where(span > 0)).clip(-1.0, 1.0)
        location = location.mask(span.eq(0), 0.0)
        volume = data['volume'].astype(float).where(data['volume'] >= 0)
        total_volume = volume.rolling(params.pressure_window).sum()
        pressure = (location * volume).rolling(params.pressure_window).sum() / total_volume.where(total_volume > 0)
        previous = close.shift(1)
        true_range = pd.concat([span, (data['high'] - previous).abs(), (data['low'] - previous).abs()], axis=1).max(axis=1, skipna=False)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame({'overnight': overnight, 'intraday': intraday, 'signature_area': area, 'signature_scale': scale, 'normalized_area': normalized_area, 'displacement': displacement, 'close_location': location, 'total_volume': total_volume, 'pressure': pressure, 'true_range': true_range, 'atr': atr}, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        features = indicators[['normalized_area', 'displacement', 'pressure', 'atr']]
        ready = features.notna().all(axis=1) & np.isfinite(features).all(axis=1)
        signature_agrees = (indicators['normalized_area'] > params.area_threshold) & (indicators['displacement'] > 0)
        pressure_agrees = indicators['pressure'] > params.pressure_threshold
        entries = (ready & signature_agrees & pressure_agrees & (indicators['atr'] > 0)).fillna(False).to_numpy(dtype=bool)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if not valid:
                active = False
                high_water, stop, age = np.nan, -np.inf, 0
            elif active:
                age += 1
                # First held close is on the bar after the entry decision.
                high_water = close if not np.isfinite(high_water) else max(high_water, close)
                stop = max(stop, high_water - params.atr_multiple * atr)
                if close <= stop or age >= params.max_hold_bars:
                    active = False
            elif entries[i]:
                active = True
                high_water, stop, age = np.nan, -np.inf, 0
            signals[i] = int(active)
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100.0, 102.0, 110.0, 109.0, 106.0, 108.0]}, index=index)
    features = pd.DataFrame({'normalized_area': [0.2, 0, 0, 0, 0, 0], 'displacement': 0.1, 'pressure': 0.3, 'atr': [1.0, 1.0, 1.0, 10.0, 10.0, 1.0]}, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams(atr_multiple=2.0)
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    assert result['size'].gt(0).all()
    params.max_hold_bars = 2
    result = strategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    features['pressure'] = 0.0
    assert strategy.generate_signals(data, features, None, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
