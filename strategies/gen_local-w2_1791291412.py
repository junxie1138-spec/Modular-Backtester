from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    complexity_window: int = 18
    entropy_cutoff: float = 0.50
    body_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local-w2_1791291412'

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        # Return lag, three-return pattern, transition lag, and confirmation.
        return int(params.complexity_window) + 5

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        window = int(params.complexity_window)
        if window < 4 or not 0.0 <= params.entropy_cutoff <= 1.0:
            raise ValueError('Invalid complexity parameters')
        if not 0.0 <= params.body_threshold <= 1.0:
            raise ValueError('Invalid body threshold')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        previous_close = close.shift(1).where(close.shift(1).gt(0.0))
        returns = (close / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out['return'] = returns
        out['return_lag_1'] = returns.shift(1)
        out['return_lag_2'] = returns.shift(2)
        a, b, c = out['return_lag_2'], out['return_lag_1'], returns
        valid_pattern = a.notna() & b.notna() & c.notna()
        # Strict comparisons break equal-return ties by chronological order.
        pattern = (
            4 * a.gt(b).astype(int)
            + 2 * a.gt(c).astype(int)
            + b.gt(c).astype(int)
        ).astype(float).where(valid_pattern)
        out['pattern'] = pattern
        out['previous_pattern'] = pattern.shift(1)
        valid_transition = pattern.notna() & out['previous_pattern'].notna()
        out['transition'] = (8 * out['previous_pattern'] + pattern).where(valid_transition)
        out['valid_transitions'] = valid_transition.astype(float).rolling(window, min_periods=window).sum()

        entropy = pd.Series(0.0, index=data.index)
        codes = (0, 1, 3, 4, 6, 7)
        for source in codes:
            marginal = (
                out['previous_pattern'].eq(source).astype(float)
                .where(valid_transition)
                .rolling(window, min_periods=window).mean()
            )
            out[f'source_probability_{source}'] = marginal
            for destination in codes:
                probability = (
                    (out['previous_pattern'].eq(source) & pattern.eq(destination))
                    .astype(float).where(valid_transition)
                    .rolling(window, min_periods=window).mean()
                )
                out[f'joint_probability_{source}_{destination}'] = probability
                conditional = probability / marginal.replace(0.0, np.nan)
                entropy -= probability * np.log(conditional.where(probability.gt(0.0), 1.0))

        # Overlapping length-three ordinal patterns have three possible successors.
        out['complexity'] = (entropy / np.log(3.0)).clip(0.0, 1.0).where(
            out['valid_transitions'].eq(window)
        )
        candle_range = (data['high'] - data['low']).where(data['high'].gt(data['low']))
        out['body_pressure'] = ((close - data['open']) / candle_range).replace(
            [np.inf, -np.inf], np.nan
        ).clip(-1.0, 1.0)
        out['regime'] = pd.Series(
            np.where(out['complexity'].le(params.entropy_cutoff), 1.0, -1.0),
            index=data.index,
        ).where(out['complexity'].notna())
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalParams,
    ) -> SignalFrame:
        pressure = indicators['body_pressure']
        ready = indicators['complexity'].notna() & pressure.notna()
        direction = pd.Series(
            np.where(pressure.gt(params.body_threshold), 1,
                     np.where(pressure.lt(-params.body_threshold), -1, 0)),
            index=data.index,
        )
        candidate = (direction * indicators['regime']).where(ready, 0).fillna(0).astype(int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)) & candidate.ne(0), 0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, (opinion, entry) in enumerate(zip(candidate.to_numpy(), confirmed.to_numpy())):
            if position != 0:
                if opinion == -position:
                    position = 0
            elif entry != 0:
                position = int(entry)
            positions[i] = position

        df = pd.DataFrame({'signal': positions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = OrdinalParams()
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame({
        'open': 100.0, 'high': 102.0, 'low': 98.0,
        'close': 101.0, 'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'complexity': 0.0,
        'regime': 1.0,
        'body_pressure': [1.0, 1.0, 0.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
    }, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0.0).all()

    index = pd.date_range('2020-01-01', periods=120)
    steps = np.arange(120, dtype=float)
    close = 100.0 * np.exp(np.cumsum(0.007 * np.sin(steps * 1.7)))
    data = pd.DataFrame({
        'open': close * np.exp(-0.004 * np.cos(steps)),
        'high': close * 1.02,
        'low': close * 0.98,
        'close': close,
        'volume': 1000.0,
    }, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:80], params)
    pd.testing.assert_frame_equal(full.iloc[:80], prefix)
    assert full['complexity'].dropna().between(0.0, 1.0).all()
    assert full['complexity'].iloc[:params.complexity_window + 3].isna().all()
    assert full['complexity'].iloc[params.complexity_window + 3:].notna().all()
    signals = strategy.generate_signals(data, full, ctx, params).data
    prefix_signals = strategy.generate_signals(data.iloc[:80], prefix, ctx, params).data
    pd.testing.assert_frame_equal(signals.iloc[:80], prefix_signals)
    assert signals['signal'].isin([-1, 0, 1]).all()


if __name__ == '__main__':
    _self_check()
