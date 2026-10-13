from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureParams:
    fractional_order: float = 0.4
    fractional_window: int = 40
    pressure_window: int = 60
    pressure_threshold: float = 0.9
    compression_window: int = 5
    reference_window: int = 20
    compression_limit: float = 0.4
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[PressureParams]):
    strategy_id = 'gen_local_1791247476'

    @classmethod
    def params_type(cls) -> type[PressureParams]:
        return PressureParams

    def warmup_bars(self, params: PressureParams) -> int:
        return int(max(
            params.fractional_window + params.pressure_window + 2,
            params.compression_window,
            params.reference_window,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: PressureParams) -> pd.DataFrame:
        if not (
            0.0 < params.fractional_order < 1.0
            and params.fractional_window >= 2
            and params.pressure_window >= 2
            and 2 <= params.compression_window < params.reference_window
            and 0.0 < params.compression_limit < 1.0
            and params.pressure_threshold > 0.0
            and params.atr_window >= 1
            and params.stop_atr > 0.0
            and params.holding_bars >= 1
        ):
            raise ValueError('Invalid strategy parameters')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        high = data['high'].astype(float).replace([np.inf, -np.inf], np.nan)
        low = data['low'].astype(float).replace([np.inf, -np.inf], np.nan)
        previous_close = close.shift(1)

        # ponytail: fixed-width fractional memory; increase the window if truncation matters.
        weights = np.ones(params.fractional_window, dtype=float)
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional = close * weights[0]
        for lag in range(1, params.fractional_window):
            fractional = fractional + close.shift(lag) * weights[lag]
        out['fractional_close'] = fractional
        reference = fractional.shift(1).rolling(
            params.pressure_window, min_periods=params.pressure_window
        )
        out['pressure_mean'] = reference.mean()
        out['pressure_std'] = reference.std(ddof=0)
        out['pressure_z'] = (
            (fractional - out['pressure_mean'])
            / out['pressure_std'].where(out['pressure_std'] > 0.0)
        )
        out['previous_pressure_z'] = out['pressure_z'].shift(1)

        out['short_high'] = high.rolling(params.compression_window).max()
        out['short_low'] = low.rolling(params.compression_window).min()
        out['reference_high'] = high.rolling(params.reference_window).max()
        out['reference_low'] = low.rolling(params.reference_window).min()
        out['short_width'] = out['short_high'] - out['short_low']
        out['reference_width'] = out['reference_high'] - out['reference_low']
        out['compression'] = (
            out['short_width']
            / out['reference_width'].where(out['reference_width'] > 0.0)
        )

        out['high_low'] = high - low
        out['high_previous_close'] = (high - previous_close).abs()
        out['low_previous_close'] = (low - previous_close).abs()
        out['true_range'] = out[
            ['high_low', 'high_previous_close', 'low_previous_close']
        ].max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PressureParams,
    ) -> SignalFrame:
        pressure_agrees = (
            indicators['pressure_z'].gt(params.pressure_threshold)
            & indicators['previous_pressure_z'].le(params.pressure_threshold)
        )
        compression_agrees = indicators['compression'].le(params.compression_limit)
        valid = pd.Series(
            np.isfinite(indicators[
                ['pressure_z', 'previous_pressure_z', 'compression', 'atr']
            ].to_numpy(dtype=float)).all(axis=1),
            index=data.index,
        )
        entry = (
            pressure_agrees & compression_agrees & valid
            & indicators['atr'].gt(0.0)
            & data['close'].gt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        stop_price = np.nan

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    entry_bar = i
                    # Freeze the stop at the next-bar opening fill and decision-bar ATR.
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]

            if active:
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - entry_bar + 1 >= params.holding_bars
                if stopped or expired:
                    active = False
                else:
                    decisions[i] = 1
            elif entry[i]:
                decisions[i] = 1
                pending = True

        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = PressureParams()
    index = pd.date_range('2020-01-01', periods=6, freq='B')
    data = pd.DataFrame({
        'open': [100.0] * 6,
        'high': [101.0] * 6,
        'low': [89.0] * 6,
        'close': [100.0, 100.0, 90.0, 100.0, 100.0, 100.0],
        'volume': [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        'pressure_z': [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        'previous_pressure_z': [np.nan, 0.0, 2.0, 0.0, 0.0, 0.0],
        'compression': [0.2] * 6,
        'atr': [1.0] * 6,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 0, 1, 0, 0, 0]
    assert result.data['size'].gt(0.0).all()
    blocked = indicators.copy()
    blocked['compression'] = 0.9
    result = GeneratedStrategy.generate_signals(None, data, blocked, None, params)
    assert result.data['signal'].eq(0).all()
    data['close'] = 100.0
    params.holding_bars = 2
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 0, 0]
