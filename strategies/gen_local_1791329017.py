from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShapeParams:
    complexity_window: int = 12
    history_window: int = 252
    min_analogs: int = 8
    complexity_width: float = 0.06
    edge_bps: float = 7.0

    def __post_init__(self):
        if self.complexity_window < 3 or self.history_window < 1:
            raise ValueError('Invalid rolling window')
        if self.min_analogs < 1 or self.min_analogs > self.history_window:
            raise ValueError('Invalid minimum analog count')
        if not np.isfinite(self.complexity_width) or self.complexity_width <= 0:
            raise ValueError('complexity_width must be finite and positive')
        if not np.isfinite(self.edge_bps) or self.edge_bps < 0:
            raise ValueError('edge_bps must be finite and nonnegative')


class GeneratedStrategy(BaseStrategy[ShapeParams]):
    strategy_id = 'gen_local_1791329017'

    @classmethod
    def params_type(cls):
        return ShapeParams

    def warmup_bars(self, params: ShapeParams) -> int:
        return params.history_window + params.complexity_window + 5

    def indicators(self, data: pd.DataFrame, params: ShapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Lehmer code for three observations; chronological order breaks ties.
        out['pattern'] = (
            2 * ((b < a).astype(int) + (c < a).astype(int))
            + (c < b).astype(int)
        ).where(valid)
        probabilities = pd.DataFrame(index=data.index)
        for code in range(6):
            occurrence = out['pattern'].eq(code).astype(float).where(valid)
            probabilities[code] = occurrence.rolling(
                params.complexity_window,
                min_periods=params.complexity_window,
            ).mean()
            out[f'probability_{code}'] = probabilities[code]
        entropy_terms = probabilities * np.log(probabilities.where(probabilities > 0, 1.0))
        out['entropy'] = -entropy_terms.sum(axis=1, min_count=6) / np.log(6.0)
        out['disequilibrium'] = (
            (probabilities - 1.0 / 6.0).pow(2).sum(axis=1, min_count=6) * 6.0 / 5.0
        )
        out['complexity'] = (out['entropy'] * out['disequilibrium']).clip(lower=0.0)
        out['complexity_bucket'] = np.floor(
            out['complexity'] / params.complexity_width
        ).clip(lower=0, upper=2)
        out['category'] = out['pattern'] * 3 + out['complexity_bucket']
        opening = data['open'].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        # At row t, the analog decision was t-3, its entry t-2, and exit t.
        # Both execution prices are already observable at the current close.
        out['matured_category'] = out['category'].shift(3)
        out['matured_return'] = (opening / opening.shift(2) - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )
        usable = out['matured_category'].notna() & out['matured_return'].notna()
        count = pd.Series(0.0, index=data.index)
        total = pd.Series(0.0, index=data.index)
        for category in range(18):
            match = out['matured_category'].eq(category) & usable
            out[f'analog_count_{category}'] = match.astype(float).rolling(
                params.history_window, min_periods=params.history_window
            ).sum()
            out[f'analog_sum_{category}'] = out['matured_return'].where(
                match, 0.0
            ).rolling(params.history_window, min_periods=params.history_window).sum()
            current = out['category'].eq(category)
            count = count + out[f'analog_count_{category}'].where(current, 0.0)
            total = total + out[f'analog_sum_{category}'].where(current, 0.0)
        out['analog_count'] = count
        out['analog_total'] = total
        out['forecast'] = (total / count.where(count > 0)).where(
            count >= params.min_analogs
        )
        out['edge'] = out['forecast'] - params.edge_bps / 10000.0
        out['ready'] = (
            pd.Series(np.arange(len(data)), index=data.index)
            >= self.warmup_bars(params) - 1
        ) & out['category'].notna() & out['forecast'].notna()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShapeParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Symmetric rule: enter and remain long exactly while this predicate holds.
        # Its reversal closes the position; there is no independent timed exit.
        entry_condition = indicators['ready'].fillna(False) & indicators['edge'].gt(0)
        df['signal'] = entry_condition.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = ShapeParams()
    strategy = GeneratedStrategy()
    n = 640
    t = np.arange(n, dtype=float)
    close = 100.0 * np.exp(0.0002 * t + 0.014 * np.sin(t * 1.7) + 0.008 * np.sin(t * 0.31))
    opening = close * np.exp(0.002 * np.cos(t * 0.83))
    data = pd.DataFrame(
        {'open': opening, 'high': np.maximum(opening, close) * 1.01,
         'low': np.minimum(opening, close) * 0.99, 'close': close, 'volume': 1000000.0},
        index=pd.date_range('2020-01-01', periods=n, freq='B'),
    )
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, None, params).data
    expected = (derived['ready'] & derived['edge'].gt(0)).astype(int).shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(signals['signal'], expected, check_names=False)
    assert signals['signal'].isin([0, 1]).all() and (signals['size'] > 0).all()
    assert signals['signal'].sum() > 0
    prefix = strategy.indicators(data.iloc[:500], params)
    pd.testing.assert_frame_equal(derived.iloc[:500], prefix)
    decision_row = 400
    assert np.isclose(
        derived['matured_return'].iloc[decision_row + 3],
        data['open'].iloc[decision_row + 3] / data['open'].iloc[decision_row + 1] - 1.0,
    )
