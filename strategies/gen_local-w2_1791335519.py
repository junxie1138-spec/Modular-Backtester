from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TideParams:
    roc_bars: int = 5
    acceleration_lag: int = 2
    scale_bars: int = 63
    atr_bars: int = 14
    entry_score: float = 0.9
    tide_gate: float = 0.5
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    hold_bars: int = 8
    min_size: float = 0.25


class GeneratedStrategy(BaseStrategy[TideParams]):
    strategy_id = 'gen_local-w2_1791335519'

    @classmethod
    def params_type(cls) -> type[TideParams]:
        return TideParams

    @staticmethod
    def warmup_bars(params: TideParams) -> int:
        return int(max(
            params.roc_bars + params.acceleration_lag + params.scale_bars + 1,
            params.atr_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: TideParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['roc'] = close.pct_change(params.roc_bars, fill_method=None)
        out['acceleration'] = out['roc'].diff(params.acceleration_lag)
        out['acceleration_scale'] = out['acceleration'].rolling(
            params.scale_bars, min_periods=params.scale_bars
        ).std(ddof=0)
        out['acceleration_z'] = out['acceleration'].div(
            out['acceleration_scale'].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out['month_phase'] = pd.Series(
            (data.index.day - 1) / data.index.days_in_month,
            index=data.index, dtype=float,
        )
        out['calendar_tide'] = np.cos(2.0 * np.pi * out['month_phase'])
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        out['score'] = (
            out['acceleration_z'].abs() * out['calendar_tide'].abs()
        ).replace([np.inf, -np.inf], np.nan)
        out['valid'] = (
            out['acceleration_z'].notna()
            & out['atr'].notna()
            & out['atr'].gt(0.0)
            & close.gt(0.0)
        )
        out.loc[out.index[:self.warmup_bars(params)], 'valid'] = False
        aligned = out['acceleration_z'] * out['calendar_tide'] > 0.0
        eligible = (
            out['valid'] & aligned
            & out['calendar_tide'].abs().ge(params.tide_gate)
            & out['score'].ge(params.entry_score)
        )
        out['entry_direction'] = np.where(
            eligible, np.sign(out['acceleration_z'].fillna(0.0)), 0
        ).astype(int)
        out['conviction'] = (
            out['score'] / (3.0 * params.entry_score)
        ).clip(0.0, 1.0).fillna(0.0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: TideParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['entry_direction'].to_numpy(dtype=int)
        conviction = indicators['conviction'].to_numpy(dtype=float)
        position = 0
        entry = np.nan
        entry_bar = -1
        best = np.nan
        stop = np.nan
        armed = False
        position_size = 1.0
        entry_atr = np.nan

        for i in range(count):
            if position:
                # The previous close's order fills at this bar's open.
                if not np.isfinite(entry):
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0.0:
                        position = 0
                        continue
                    entry_bar = i
                    best = position * entry
                    stop = best - params.trail_atr * entry_atr

                price = closes[i]
                if not np.isfinite(price) or price <= 0.0:
                    position = 0
                    entry = np.nan
                    continue

                # Favorable coordinates increase for both longs and shorts.
                # Ratcheting this stop upward tightens either direction.
                favorable_price = position * price
                best = max(best, favorable_price)
                if best - position * entry >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, position * entry)
                    if np.isfinite(atr[i]) and atr[i] > 0.0:
                        stop = max(stop, best - params.trail_atr * atr[i])

                # Close-confirmed exits become flat at the following open.
                if favorable_price <= stop or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
                    entry = np.nan
                    continue

            elif candidates[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                position = int(candidates[i])
                entry = np.nan
                entry_atr = atr[i]
                armed = False
                position_size = float(np.clip(
                    params.min_size + (1.0 - params.min_size) * conviction[i],
                    params.min_size, 1.0,
                ))

            if position:
                signals[i] = position
                sizes[i] = position_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = TideParams(trail_atr=1.5)
    for direction in (1, -1):
        index = pd.date_range('2020-01-01', periods=7)
        closes = 100.0 + direction * np.array([0, 0, 2, 3, 1, 0, 0], dtype=float)
        data = pd.DataFrame({'open': 100.0, 'close': closes}, index=index)
        indicators = pd.DataFrame({
            'atr': 1.0,
            'entry_direction': [direction, 0, 0, 0, 0, 0, 0],
            'conviction': 0.5,
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, direction, direction, 0, 0]
        assert np.allclose(result.loc[index[1:5], 'size'], 0.625)
        assert result['size'].gt(0.0).all()


if __name__ == '__main__':
    _self_check()
