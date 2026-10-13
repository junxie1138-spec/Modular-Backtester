from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 12
    visibility_min: float = 0.6
    pressure_window: int = 3
    pressure_min: float = 0.15
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = 'gen_local-w2_1790967710'

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.graph_window + 2, params.pressure_window, 4))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_window < 4 or params.pressure_window < 1:
            raise ValueError('Invalid indicator window')
        if not 0.0 <= params.visibility_min <= 1.0:
            raise ValueError('visibility_min must be between zero and one')
        if not 0.0 <= params.pressure_min <= 1.0:
            raise ValueError('pressure_min must be between zero and one')
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        log_close = np.log(close.where(close > 0.0))
        nearest_min = pd.Series(np.inf, index=data.index, dtype=float)
        long_edges = pd.Series(0.0, index=data.index)
        valid = log_close.notna()
        # An older node is visible when its backward slope is a strict minimum.
        for distance in range(1, params.graph_window + 1):
            slope = (log_close - log_close.shift(distance)) / distance
            valid &= slope.notna()
            visible = slope.lt(nearest_min) & slope.notna()
            if distance >= 4:
                long_edges += visible.astype(float)
            nearest_min = np.minimum(nearest_min, slope)
        span = (long_edges / (params.graph_window - 3)).where(valid)
        volume = data['volume'].astype(float).replace([np.inf, -np.inf], np.nan)
        volume = volume.where(volume >= 0.0)
        day_range = (data['high'] - data['low']).replace([np.inf, -np.inf], np.nan)
        body = (data['close'] - data['open']).replace([np.inf, -np.inf], np.nan)
        usable = day_range.gt(0.0) & body.abs().le(day_range)
        numerator = (volume * body.where(usable)).rolling(params.pressure_window).sum()
        denominator = (volume * day_range.where(usable)).rolling(params.pressure_window).sum()
        pressure = numerator / denominator.where(denominator > 0.0)
        out = pd.DataFrame(index=data.index)
        out['visibility_span'] = span
        out['visibility_expansion'] = span.diff()
        out['momentum'] = log_close.diff(3)
        out['buying_pressure'] = pressure
        out['entry'] = (
            out['visibility_span'].ge(params.visibility_min)
            & out['visibility_expansion'].gt(0.0)
            & out['momentum'].gt(0.0)
            & out['buying_pressure'].ge(params.pressure_min)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        if params.max_hold_bars not in (1, 2):
            raise ValueError('max_hold_bars must be one or two')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError('profit_target must be positive and finite')
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = 0.0
        held = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry_price = opens[i]
                    active = True
                    held = 0
            if active:
                held += 1
                # Close-observed target: the exit fills at the following open.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held >= params.max_hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range('2020-01-01', periods=4)
    data = pd.DataFrame({'open': 100.0, 'high': 103.0, 'low': 99.0,
                         'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 0]
    data.loc[index[1], 'close'] = 102.0
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 0, 0]
    index = pd.date_range('2020-01-01', periods=40)
    price = np.exp(4.0 + 0.0001 * np.arange(40, dtype=float) ** 2)
    data = pd.DataFrame({'open': price * 0.999, 'high': price * 1.002,
                         'low': price * 0.998, 'close': price, 'volume': 1000.0}, index=index)
    full = strategy.indicators(data, params)
    assert np.isclose(full['visibility_span'].iloc[-1], 1.0)
    prefix = strategy.indicators(data.iloc[:25], params)
    pd.testing.assert_frame_equal(full.iloc[:25], prefix)


if __name__ == '__main__':
    _self_check()
