from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CostBasisParams:
    basis_window: int = 30
    level_span: int = 5
    slope_span: int = 8
    innovation_window: int = 20
    innovation_threshold: float = 0.35
    atr_window: int = 14
    atr_multiple: float = 3.0
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[CostBasisParams]):
    strategy_id = 'gen_local-w2_1791259113'

    @classmethod
    def params_type(cls) -> type[CostBasisParams]:
        return CostBasisParams

    def warmup_bars(self, params: CostBasisParams) -> int:
        return int(max(200, params.atr_window + 1,
                       params.basis_window + params.level_span
                       + max(params.slope_span, params.innovation_window) + 2))

    def indicators(self, data: pd.DataFrame, params: CostBasisParams) -> pd.DataFrame:
        if min(params.basis_window, params.level_span, params.slope_span,
               params.innovation_window, params.atr_window) < 2:
            raise ValueError('Indicator periods must be at least two.')
        if params.atr_multiple <= 0 or params.max_hold_bars < 1:
            raise ValueError('Stop distance and holding period must be positive.')
        if params.innovation_threshold < 0:
            raise ValueError('Innovation threshold must be nonnegative.')
        close = data['close'].astype(float)
        volume = data['volume'].astype(float).clip(lower=0)
        total = volume.rolling(params.basis_window).sum().replace(0, np.nan)
        basis = (close * volume).rolling(params.basis_window).sum() / total
        second = (close.pow(2) * volume).rolling(params.basis_window).sum() / total
        spread = (second - basis.pow(2)).clip(lower=0).pow(0.5).replace(0, np.nan)
        position = ((close - basis) / spread).clip(-4.0, 4.0)
        level = position.ewm(span=params.level_span, adjust=False,
                             min_periods=params.level_span).mean()
        change = level.diff()
        slope = change.ewm(span=params.slope_span, adjust=False,
                           min_periods=params.slope_span).mean()
        predicted_slope = slope.shift(1)
        noise = change.rolling(params.innovation_window).std(ddof=0).shift(1).replace(0, np.nan)
        innovation = (change - predicted_slope) / noise
        previous = close.shift(1)
        true_range = pd.concat([(data['high'] - data['low']).abs(),
                                (data['high'] - previous).abs(),
                                (data['low'] - previous).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame({
            'basis': basis,
            'basis_change': basis.diff(),
            'basis_second_moment': second,
            'basis_spread': spread,
            'relative_position': position,
            'latent_level': level,
            'latent_change': change,
            'latent_slope': slope,
            'predicted_slope': predicted_slope,
            'innovation_noise': noise,
            'slope_innovation': innovation,
            'true_range': true_range,
            'atr': atr,
            'ma200': close.rolling(200).mean(),
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CostBasisParams) -> SignalFrame:
        columns = ['relative_position', 'latent_slope', 'slope_innovation',
                   'basis_change', 'atr', 'ma200']
        values = indicators[columns].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        trailing_floor = -np.inf
        short_ceiling = np.inf
        for i in range(len(data)):
            close = closes[i]
            position, slope, innovation, migration, atr, ma = values[i]
            if not np.isfinite(close) or not np.isfinite(values[i]).all() or atr <= 0:
                side = 0
                age = 0
                continue
            if side:
                age += 1
                # Watermarks start on the first close after the entry decision.
                high_water = close if np.isnan(high_water) else max(high_water, close)
                trailing_floor = max(trailing_floor, high_water - params.atr_multiple * atr)
                low_water = close if np.isnan(low_water) else min(low_water, close)
                short_ceiling = min(short_ceiling, low_water + params.atr_multiple * atr)
                # Both directions obey the required high-water drawdown exit.
                # Shorts additionally have a mirrored adverse-move trailing exit.
                stopped = close <= trailing_floor or (side == -1 and close >= short_ceiling)
                regime_broken = (side == 1 and close <= ma) or (side == -1 and close >= ma)
                if stopped or regime_broken or age >= params.max_hold_bars:
                    side = 0
                    age = 0
                    continue
            else:
                long_entry = (close > ma and migration > 0 and slope > 0
                              and -1.5 < position < -0.15
                              and innovation > params.innovation_threshold)
                short_entry = (close < ma and migration < 0 and slope < 0
                               and 0.15 < position < 1.5
                               and innovation < -params.innovation_threshold)
                side = 1 if long_entry else (-1 if short_entry else 0)
                if side:
                    age = 0
                    high_water = np.nan
                    low_water = np.nan
                    trailing_floor = -np.inf
                    short_ceiling = np.inf
            signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'open': [100, 100, 105, 100, 98],
                         'high': [101, 106, 106, 101, 99],
                         'low': [99, 99, 99, 97, 97],
                         'close': [100, 105, 100, 98, 98],
                         'volume': 1000}, index=index)
    ind = pd.DataFrame({'relative_position': -0.5, 'latent_slope': 1.0,
                        'slope_innovation': 1.0, 'basis_change': 1.0,
                        'atr': [2, 2, 20, 2, 2], 'ma200': 50.0}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, CostBasisParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
