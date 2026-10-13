from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    depth_window: int = 20
    analog_window: int = 120
    neighbors: int = 5
    percentile_window: int = 126
    depth_percentile: float = 0.70
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = 'gen_local-w2_1790940608'

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return int(max(
            params.depth_window + params.analog_window + 6,
            params.depth_window + params.percentile_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not (
            params.depth_window >= 2
            and params.analog_window >= params.neighbors >= 1
            and params.percentile_window >= 2
            and 0.0 < params.depth_percentile < 1.0
            and params.atr_window >= 1
            and params.breakeven_pct > 0.0
            and params.trail_atr > 0.0
        ):
            raise ValueError('Invalid strategy parameters')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0.0)
        peak = close.rolling(params.depth_window).max()
        out['depth'] = 1.0 - close / peak
        out['depth_lag1'] = out['depth'].shift(1)
        out['depth_lag2'] = out['depth'].shift(2)
        out['depth_cutoff'] = out['depth'].shift(1).rolling(
            params.percentile_window
        ).quantile(params.depth_percentile)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = true_range.rolling(params.atr_window).mean()
        shape = out[['depth', 'depth_lag1', 'depth_lag2']]
        # An episode beginning after shape bar j ends at close j+2.
        completed_return = close / data['open'].shift(1).where(
            data['open'].shift(1) > 0.0
        ) - 1.0
        distances = np.full((len(data), params.analog_window), np.inf)
        outcomes = np.full_like(distances, np.nan)
        # Historical shapes and both outcome sessions precede the query shape.
        for column, lag in enumerate(range(5, params.analog_window + 5)):
            delta = shape - shape.shift(lag)
            distance = delta.pow(2).sum(axis=1, min_count=3).to_numpy()
            outcome = completed_return.shift(lag - 2).to_numpy()
            valid = np.isfinite(distance) & np.isfinite(outcome)
            distances[valid, column] = distance[valid]
            outcomes[valid, column] = outcome[valid]
        nearest = np.argpartition(
            distances, params.neighbors - 1, axis=1
        )[:, :params.neighbors]
        matched_distances = np.take_along_axis(distances, nearest, axis=1)
        matched_outcomes = np.take_along_axis(outcomes, nearest, axis=1)
        enough = np.isfinite(matched_distances).all(axis=1)
        forecast = np.full(len(data), np.nan)
        forecast[enough] = matched_outcomes[enough].mean(axis=1)
        out['forecast'] = forecast
        eligible = (
            out['depth'].gt(out['depth_cutoff'])
            & out['forecast'].notna()
            & out['atr'].gt(0.0)
        )
        out['candidate'] = np.where(
            eligible, np.sign(out['forecast'].fillna(0.0)), 0
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        position = 0
        entry = 0.0
        age = 0
        armed = False
        stop = -np.inf
        for i in range(len(data)):
            # A previous close decision starts a position at this bar's open.
            if position == 0 and i > 0 and decisions[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    position = int(decisions[i - 1])
                    entry = opens[i]
                    age = 0
                    armed = False
                    stop = -np.inf
            if position != 0:
                age += 1
                valid_close = np.isfinite(closes[i]) and closes[i] > 0.0
                signed_close = position * closes[i]
                exit_now = age >= 2
                if valid_close:
                    # Stops are close-observed; exit orders fill next open.
                    if armed and signed_close <= stop:
                        exit_now = True
                    if not exit_now:
                        gain = position * (closes[i] / entry - 1.0)
                        if gain >= params.breakeven_pct:
                            armed = True
                        if armed:
                            stop = max(stop, position * entry)
                            if np.isfinite(atrs[i]) and atrs[i] > 0.0:
                                # Profit-coordinate stop only rises; short price stops fall.
                                stop = max(
                                    stop, signed_close - params.trail_atr * atrs[i]
                                )
                if exit_now:
                    position = 0
                    decisions[i] = 0
                else:
                    decisions[i] = position
            elif np.isfinite(closes[i]) and closes[i] > 0.0:
                decisions[i] = candidates[i]
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=360, freq='B')
    t = np.arange(len(index), dtype=float)
    close = 100.0 + 0.025 * t + 3.0 * np.sin(t / 5.0)
    data = pd.DataFrame({
        'open': close + 0.1 * np.cos(t),
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close,
        'volume': np.full(len(index), 1000000.0),
    }, index=index)
    strategy = GeneratedStrategy()
    params = AnalogParams()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:300], params)
    pd.testing.assert_frame_equal(full.iloc[:300], prefix)
    signals = strategy.generate_signals(data, full, None, params).data
    earlier = strategy.generate_signals(data.iloc[:300], prefix, None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:300], earlier)
    assert signals['signal'].isin([-1, 0, 1]).all()
    assert signals['size'].gt(0.0).all()
    assert signals['signal'].iloc[0] == 0
    assert full['candidate'].abs().sum() > 0
    # An isolated decision fills one bar later and expires after two sessions.
    manual = pd.DataFrame({'candidate': [1, 0, 0, 0, 0], 'atr': 1.0}, index=index[:5])
    flat = data.iloc[:5].copy()
    flat[['open', 'close']] = 100.0
    result = strategy.generate_signals(flat, manual, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    # Both directions activate breakeven and exit after a close crosses it.
    for direction in (1, -1):
        manual['candidate'] = [direction, 0, 0, 0, 0]
        flat['close'] = [100.0, 100.0 + direction, 100.0 - direction, 100.0, 100.0]
        result = strategy.generate_signals(flat, manual, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, 0, 0]


if __name__ == '__main__':
    _self_check()
