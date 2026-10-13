from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    frac_order: float = 0.45
    frac_terms: int = 32
    shape_bars: int = 16
    analog_lookback: int = 252
    neighbors: int = 5
    max_distance: float = 0.75
    edge_threshold: float = 0.006
    flow_threshold: float = 0.08
    atr_bars: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local-w2_1791265489'

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    @staticmethod
    def warmup_bars(params: ShapeParams) -> int:
        return int(max(
            params.frac_terms + params.shape_bars + params.analog_lookback,
            params.atr_bars + 1,
            params.hold_bars + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        p = params
        if not (
            0.0 < p.frac_order < 1.0
            and p.frac_terms >= 2
            and p.shape_bars >= 4
            and p.hold_bars >= 1
            and p.atr_bars >= 1
            and p.neighbors >= 1
            and p.analog_lookback >= p.hold_bars + p.shape_bars + p.neighbors - 1
            and p.max_distance > 0
            and p.edge_threshold > 0
            and 0 <= p.flow_threshold < 1
            and p.stop_atr > 0
        ):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        weights = np.ones(p.frac_terms, dtype=float)
        for j in range(1, p.frac_terms):
            weights[j] = -weights[j - 1] * (p.frac_order - j + 1) / j
        # ponytail: finite fractional memory; increase frac_terms if truncation matters.
        fractional = pd.Series(0.0, index=data.index)
        for j, weight in enumerate(weights):
            fractional = fractional + weight * close.shift(j)
        out['fractional_close'] = fractional
        out['shape_mean'] = fractional.rolling(p.shape_bars).mean()
        out['shape_std'] = fractional.rolling(p.shape_bars).std(ddof=0).replace(0.0, np.nan)
        feature_names = []
        for j, offset in enumerate(np.linspace(0, p.shape_bars - 1, 4).astype(int)):
            name = 'shape_' + str(j)
            out[name] = (fractional.shift(int(offset)) - out['shape_mean']) / out['shape_std']
            feature_names.append(name)
        previous = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(p.atr_bars).mean()
        span = (data['high'] - data['low']).replace(0.0, np.nan)
        out['close_location'] = ((2.0 * close - data['high'] - data['low']) / span).clip(-1.0, 1.0).fillna(0.0)
        out['signed_volume'] = out['close_location'] * data['volume']
        out['volume_sum'] = data['volume'].rolling(p.hold_bars).sum()
        out['signed_volume_sum'] = out['signed_volume'].rolling(p.hold_bars).sum()
        out['flow'] = out['signed_volume_sum'] / out['volume_sum'].replace(0.0, np.nan)
        out['realized_return'] = close / close.shift(p.hold_bars).replace(0.0, np.nan) - 1.0
        n = len(data)
        best_distance = np.full((n, p.neighbors), np.inf)
        best_outcome = np.zeros((n, p.neighbors))
        features = out[feature_names]
        # Every example's outcome ends before the current shape starts.
        for lag in range(p.hold_bars + p.shape_bars, p.analog_lookback + 1):
            distance = np.sqrt(((features - features.shift(lag)) ** 2).mean(axis=1, skipna=False)).to_numpy()
            outcome = out['realized_return'].shift(lag - p.hold_bars).to_numpy()
            valid = np.isfinite(distance) & np.isfinite(outcome) & (distance <= p.max_distance)
            worst = best_distance.argmax(axis=1)
            rows = np.arange(n)
            replace = valid & (distance < best_distance[rows, worst])
            selected = rows[replace]
            best_distance[selected, worst[replace]] = distance[replace]
            best_outcome[selected, worst[replace]] = outcome[replace]
        out['analog_count'] = np.isfinite(best_distance).sum(axis=1)
        out['analog_edge'] = np.where(
            out['analog_count'].to_numpy() == p.neighbors,
            best_outcome.mean(axis=1),
            np.nan,
        )
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        p = params
        long_entry = (indicators['analog_edge'] > p.edge_threshold) & (indicators['flow'] > p.flow_threshold)
        short_entry = (indicators['analog_edge'] < -p.edge_threshold) & (indicators['flow'] < -p.flow_threshold)
        desired = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending = False
        entry_bar = 0
        stop = np.nan
        entry_atr = np.nan
        for i in range(len(data)):
            if position and pending:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    position = 0
                    pending = False
                    continue
                entry_bar = i
                stop = opens[i] - position * p.stop_atr * entry_atr
                pending = False
            if position:
                stopped = np.isfinite(closes[i]) and (
                    (position == 1 and closes[i] < stop)
                    or (position == -1 and closes[i] > stop)
                )
                expired = i - entry_bar + 1 >= p.hold_bars
                if stopped or expired:
                    position = 0
                signals[i] = position
                continue
            if desired[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(desired[i])
                entry_atr = atr[i]
                pending = True
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    p = ShapeParams()
    index = pd.date_range('2020-01-01', periods=6)
    for direction in (1, -1):
        close = np.array([100.0, 100.0, 100.0 - direction * 5.0, 100.0, 100.0, 100.0])
        data = pd.DataFrame({'open': 100.0, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'analog_edge': direction * 0.02, 'flow': direction * 0.2, 'atr': 1.0}, index=index)
        result = GeneratedStrategy.generate_signals(data, ind, None, p).data
        assert result['signal'].iloc[:4].tolist() == [0, direction, direction, 0]
        ind['flow'] = -direction * 0.2
        assert GeneratedStrategy.generate_signals(data, ind, None, p).data['signal'].eq(0).all()
    index = pd.date_range('2020-01-01', periods=24)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    ind = pd.DataFrame({'analog_edge': 0.02, 'flow': 0.2, 'atr': 1.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, ind, None, p).data
    assert result['signal'].iloc[1:19].eq(1).all()
    assert result['signal'].iloc[19] == 0
    assert result['size'].gt(0).all()
    derived = GeneratedStrategy.indicators(data, p)
    assert derived.index.equals(data.index)
    assert derived['analog_edge'].isna().all()


if __name__ == '__main__':
    _self_check()
