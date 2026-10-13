from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    rank_window: int = 60
    analog_window: int = 120
    distance_limit: float = 0.12
    min_analogs: int = 8
    agreement_threshold: float = 0.60
    forecast_threshold: float = 0.0015
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1791340495'

    @classmethod
    def params_type(cls) -> type[ShapeParams]:
        return ShapeParams

    @staticmethod
    def warmup_bars(params: ShapeParams) -> int:
        return int(max(200, params.atr_window + params.rank_window + params.analog_window + 5))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        span = (data['high'] - data['low']).replace(0.0, np.nan)
        out['close_location'] = (close - data['low']) / span
        out['body_pressure'] = (close - data['open']) / span
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['relative_range'] = span / out['atr'].replace(0.0, np.nan)
        for source, target in (
            ('close_location', 'location_rank'),
            ('body_pressure', 'body_rank'),
            ('relative_range', 'range_rank'),
        ):
            out[target] = out[source].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['previous_location_rank'] = out['location_rank'].shift(1)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        features = out[['previous_location_rank', 'location_rank', 'body_rank', 'range_rank']]
        count = pd.Series(0.0, index=data.index)
        total = pd.Series(0.0, index=data.index)
        positive = pd.Series(0.0, index=data.index)
        negative = pd.Series(0.0, index=data.index)
        # Every analog's two-session outcome ends at least two bars before today.
        for lag in range(4, params.analog_window + 4):
            historical = features.shift(lag)
            distance = (features - historical).abs().mean(axis=1, skipna=False)
            outcome = close.shift(lag - 2) / close.shift(lag).replace(0.0, np.nan) - 1.0
            match = distance.le(params.distance_limit) & outcome.notna() & np.isfinite(outcome)
            count += match.astype(float)
            total += outcome.where(match, 0.0)
            positive += (match & outcome.gt(0.0)).astype(float)
            negative += (match & outcome.lt(0.0)).astype(float)
        denominator = count.replace(0.0, np.nan)
        out['analog_count'] = count
        out['forecast'] = total / denominator
        out['agreement'] = positive.where(out['forecast'].ge(0.0), negative) / denominator
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShapeParams) -> SignalFrame:
        usable = (
            indicators['analog_count'].ge(params.min_analogs)
            & indicators['agreement'].ge(params.agreement_threshold)
            & indicators['atr'].gt(0.0)
            & indicators['ma200'].notna()
        )
        long_entry = usable & indicators['forecast'].gt(params.forecast_threshold) & data['close'].gt(indicators['ma200'])
        short_entry = usable & indicators['forecast'].lt(-params.forecast_threshold) & data['close'].lt(indicators['ma200'])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        direction = 0
        entry = 0.0
        stop = -np.inf
        armed = False
        age = 0
        for i in range(len(data)):
            if direction == 0 and i > 0 and raw[i - 1] != 0:
                direction = int(raw[i - 1])
                entry = opens[i]
                age = 0
                armed = False
                if not np.isfinite(entry) or entry <= 0.0 or not np.isfinite(atr[i - 1]):
                    direction = 0
                    continue
                # Signed stops ratchet upward for either direction.
                stop = direction * entry - params.trail_atr * atr[i - 1]
            if direction != 0:
                age += 1
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]):
                    direction = 0
                    continue
                signed_close = direction * closes[i]
                exit_now = signed_close <= stop
                if not exit_now:
                    profit = direction * (closes[i] - entry) / entry
                    armed = armed or profit >= params.breakeven_pct
                    if armed:
                        stop = max(stop, direction * entry, signed_close - params.trail_atr * atr[i])
                        exit_now = signed_close <= stop
                # Stops are evaluated at closes; exits execute on the next bar.
                if exit_now or age >= 2:
                    direction = 0
                    continue
                raw[i] = direction
            else:
                raw[i] = int(candidates[i])
        df = pd.DataFrame({'signal': raw, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({
        'open': [100.0, 100.0, 102.0, 100.5],
        'high': [101.0, 103.0, 103.0, 102.0],
        'low': [99.0, 99.0, 100.0, 100.0],
        'close': [100.0, 102.0, 100.5, 101.0],
        'volume': [1000.0] * 4,
    }, index=index)
    indicators = pd.DataFrame({
        'analog_count': [10.0] * 4,
        'agreement': [0.8] * 4,
        'atr': [1.0] * 4,
        'ma200': [90.0] * 4,
        'forecast': [0.01, 0.0, 0.0, 0.0],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, ShapeParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0]
    assert result['size'].gt(0.0).all()
    data.loc[index[1], 'close'] = 98.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, ShapeParams()).data
    assert result['signal'].tolist() == [0, 1, 0, 0]
    assert GeneratedStrategy.warmup_bars(ShapeParams()) >= 200


if __name__ == '__main__':
    _self_check()
