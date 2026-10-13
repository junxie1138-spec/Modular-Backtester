from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalMemoryParams:
    fractional_order: float = 0.45
    fractional_window: int = 24
    correlation_window: int = 45
    entry_strength: float = 0.25
    min_partial_correlation: float = 0.08
    atr_window: int = 14
    trail_atr: float = 1.25
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[FractionalMemoryParams]):
    strategy_id = 'gen_local-w2_1791236828'

    @classmethod
    def params_type(cls) -> type[FractionalMemoryParams]:
        return FractionalMemoryParams

    @staticmethod
    def warmup_bars(params: FractionalMemoryParams) -> int:
        return int(max(params.fractional_window + params.correlation_window + 3,
                       params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: FractionalMemoryParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError('fractional_order must be between zero and one')
        if params.fractional_window < 2 or params.correlation_window < 5:
            raise ValueError('indicator windows are too short')
        if params.atr_window < 1 or params.trail_atr <= 0.0:
            raise ValueError('ATR settings must be positive')
        if params.entry_strength <= 0.0:
            raise ValueError('entry_strength must be positive')
        if not (0.0 <= params.min_partial_correlation < 1.0):
            raise ValueError('invalid partial correlation threshold')
        if params.holding_bars not in (1, 2):
            raise ValueError('holding_bars must be one or two')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        fractional_close = close.copy()
        weight = 1.0
        # ponytail: finite fractional memory; increase the window if truncation matters.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional_close = fractional_close + weight * close.shift(lag)
        out['fractional_close'] = fractional_close
        out['innovation'] = fractional_close.diff()
        out['lag_one'] = out['innovation'].shift(1)
        out['lag_two'] = out['innovation'].shift(2)
        window = params.correlation_window
        out['rho_one'] = out['innovation'].rolling(window).corr(out['lag_one'])
        out['rho_two'] = out['innovation'].rolling(window).corr(out['lag_two'])
        out['conditional_denominator'] = (1.0 - out['rho_one'].pow(2)).where(
            1.0 - out['rho_one'].pow(2) > 1e-6)
        out['partial_two'] = (
            (out['rho_two'] - out['rho_one'].pow(2)) /
            out['conditional_denominator']
        ).clip(-0.95, 0.95)
        out['coefficient_one'] = out['rho_one'] * (1.0 - out['partial_two'])
        out['innovation_scale'] = out['innovation'].rolling(window).std(ddof=0)
        out['forecast'] = (
            out['coefficient_one'] * out['innovation'] +
            out['partial_two'] * out['lag_one']
        ) / out['innovation_scale'].where(out['innovation_scale'] > 1e-12)
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalMemoryParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.20, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        forecasts = indicators['forecast'].to_numpy(dtype=float)
        partials = indicators['partial_two'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        position = 0
        age = 0
        high_water = 0.0
        stop = -np.inf
        entry_size = 0.20

        for i in range(len(data)):
            close, forecast, partial, atr = closes[i], forecasts[i], partials[i], atrs[i]
            valid = bool(np.isfinite([close, forecast, partial, atr]).all())
            if not valid or close <= 0.0 or atr <= 0.0:
                position = 0
                age = 0
                continue

            if position:
                age += 1
                high_water = max(high_water, close)
                stop = max(stop, high_water - params.trail_atr * atr)
                # Required high-water pullback exit applies to both directions;
                # for shorts it harvests a downward move rather than protecting it.
                if close <= stop or age >= params.holding_bars:
                    position = 0
                    age = 0
                    continue
                signals[i] = position
                sizes[i] = entry_size
                continue

            if (abs(forecast) >= params.entry_strength and
                    abs(partial) >= params.min_partial_correlation):
                position = 1 if forecast > 0.0 else -1
                age = 0
                high_water = close
                stop = close - params.trail_atr * atr
                entry_size = float(np.clip(
                    0.20 + 0.75 * abs(forecast) / (3.0 * params.entry_strength),
                    0.20, 0.95))
                signals[i] = position
                sizes[i] = entry_size

        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.20).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = FractionalMemoryParams()
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({
        'open': [100.0, 100.0, 98.0, 98.0],
        'high': [101.0, 101.0, 99.0, 99.0],
        'low': [99.0, 97.0, 97.0, 97.0],
        'close': [100.0, 98.0, 98.0, 98.0],
        'volume': [1000.0] * 4,
    }, index=index)
    features = pd.DataFrame({
        'forecast': [0.75, 0.0, 0.0, 0.0],
        'partial_two': [0.30] * 4,
        'atr': [1.0] * 4,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0]
    assert result['size'].gt(0.0).all()
    features['forecast'] = [-0.75, 0.0, 0.0, 0.0]
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, -1, 0, 0]
    data['close'] = [100.0, 101.0, 102.0, 103.0]
    features['forecast'] = [0.75, 0.0, 0.0, 0.0]
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    features['forecast'] = np.nan
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result['signal'].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 72


if __name__ == '__main__':
    _self_check()
