from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OccupancyParams:
    range_days: int = 20
    slope_span: int = 8
    innovation_span: int = 12
    innovation_threshold: float = 1.2
    atr_days: int = 14
    trail_atr: float = 1.6
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[OccupancyParams]):
    strategy_id = 'gen_local-w2_1791258876'

    @classmethod
    def params_type(cls) -> type[OccupancyParams]:
        return OccupancyParams

    def warmup_bars(self, params: OccupancyParams) -> int:
        return int(max(200, params.atr_days + 1,
                       params.range_days + params.slope_span
                       + params.innovation_span + 3))

    def indicators(self, data: pd.DataFrame,
                   params: OccupancyParams) -> pd.DataFrame:
        p = params
        close = data['close'].astype(float)
        out = pd.DataFrame(index=data.index)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['range_low'] = data['low'].rolling(
            p.range_days, min_periods=p.range_days).min().shift(1)
        out['range_high'] = data['high'].rolling(
            p.range_days, min_periods=p.range_days).max().shift(1)
        out['range_width'] = out['range_high'] - out['range_low']
        out['occupancy'] = ((close - out['range_low'])
                            / out['range_width'].where(out['range_width'] > 0))
        bounded = out['occupancy'].clip(0.01, 0.99)
        out['log_odds'] = np.log(bounded / (1.0 - bounded))
        out['velocity'] = out['log_odds'].diff()
        out['latent_slope'] = out['velocity'].ewm(
            span=p.slope_span, adjust=False,
            min_periods=p.slope_span).mean()
        out['innovation'] = out['velocity'] - out['latent_slope'].shift(1)
        out['innovation_scale'] = out['innovation'].pow(2).ewm(
            span=p.innovation_span, adjust=False,
            min_periods=p.innovation_span).mean().pow(0.5).shift(1)
        out['innovation_score'] = (out['innovation']
            / out['innovation_scale'].where(out['innovation_scale'] > 1e-12))
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(
            p.atr_days, min_periods=p.atr_days).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame,
                         indicators: pd.DataFrame,
                         ctx: StrategyContext,
                         params: OccupancyParams) -> SignalFrame:
        p = params
        if (min(p.range_days, p.slope_span, p.innovation_span,
                p.atr_days, p.hold_days) < 1
                or p.trail_atr <= 0 or p.innovation_threshold <= 0):
            raise ValueError('Lookbacks, holding period and thresholds must be positive')
        close = data['close'].to_numpy(dtype=float)
        features = indicators[['ma200', 'occupancy', 'latent_slope',
                               'innovation_score', 'atr']].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest = lowest = high_stop = short_stop = np.nan
        for i, price in enumerate(close):
            ma, occupancy, slope, shock, atr = features[i]
            valid = np.isfinite(price) and np.isfinite(features[i]).all() and atr > 0
            if not valid:
                position = 0
                age = 0
                continue
            if position:
                age += 1
                highest = max(highest, price)
                lowest = min(lowest, price)
                # Preserve the required highest-close drawdown exit for both sides.
                high_stop = max(high_stop, highest - p.trail_atr * atr)
                # Shorts additionally have a mirrored adverse-move trailing stop.
                short_stop = min(short_stop, lowest + p.trail_atr * atr)
                regime_exit = (position == 1 and price <= ma) or (
                    position == -1 and price >= ma)
                if (price <= high_stop or age >= p.hold_days or regime_exit
                        or (position == -1 and price >= short_stop)):
                    position = 0
                    age = 0
                signals[i] = position
                continue
            if (price > ma and 0.45 <= occupancy <= 0.75
                    and slope > 0 and shock >= p.innovation_threshold):
                position = 1
            elif (price < ma and 0.25 <= occupancy <= 0.55
                    and slope < 0 and shock <= -p.innovation_threshold):
                position = -1
            if position:
                age = 0
                highest = lowest = price
                high_stop = price - p.trail_atr * atr
                short_stop = price + p.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'close': [100., 101., 103., 102., 99., 100., 100.]},
                        index=index)
    features = pd.DataFrame({
        'ma200': 90., 'occupancy': 0.6, 'latent_slope': 0.1,
        'innovation_score': [2., 0., 0., 0., 0., 0., 0.],
        'atr': [1., 1., 1., 10., 10., 10., 10.]
    }, index=index)
    result = GeneratedStrategy().generate_signals(
        data, features, None, OccupancyParams(hold_days=5))
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result.data.index.equals(index)
    assert (result.data['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
