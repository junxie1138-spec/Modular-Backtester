from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    peak_window: int = 60
    ordinal_window: int = 18
    comparison_lag: int = 5
    atr_window: int = 14
    min_drawdown: float = 0.02
    max_drawdown: float = 0.12
    descending_threshold: float = 0.35
    complexity_rise: float = 0.015
    trail_atr: float = 2.0
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791305504'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(
            params.peak_window,
            params.atr_window + 1,
            params.ordinal_window + 2 + params.comparison_lag + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.peak_window, params.ordinal_window, params.atr_window) < 2:
            raise ValueError('Rolling windows must be at least two bars.')
        if params.comparison_lag < 1 or params.max_hold_bars < 1:
            raise ValueError('Comparison lag and holding limit must be positive.')
        if not 0 <= params.min_drawdown < params.max_drawdown < 1:
            raise ValueError('Invalid drawdown bounds.')
        if not 0 <= params.descending_threshold <= 1:
            raise ValueError('Invalid descending-pattern threshold.')
        if not np.isfinite(params.complexity_rise) or params.complexity_rise < 0:
            raise ValueError('Complexity rise must be finite and nonnegative.')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('ATR multiplier must be finite and positive.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['previous_close'] = close.shift(1)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = 1.0 - close / out['peak'].where(out['peak'] > 0)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        valid_bar = np.isfinite(data[['high', 'low', 'close']]).all(axis=1)
        out['true_range'] = out['true_range'].where(valid_bar & out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()

        # Pairwise comparisons encode six three-close ordinal permutations.
        # Equal prices retain chronological order, giving deterministic ties.
        oldest = close.shift(2)
        middle = close.shift(1)
        valid_pattern = np.isfinite(oldest) & np.isfinite(middle) & np.isfinite(close)
        out['ordinal_code'] = (
            4 * (oldest > middle).astype(int)
            + 2 * (oldest > close).astype(int)
            + (middle > close).astype(int)
        ).astype(float).where(valid_pattern)
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            column = f'probability_{code}'
            probability_columns.append(column)
            out[column] = out['ordinal_code'].eq(code).astype(float).where(
                valid_pattern
            ).rolling(params.ordinal_window, min_periods=params.ordinal_window).mean()

        probabilities = out[probability_columns]
        log_probabilities = np.log(probabilities.where(probabilities > 0, 1.0))
        out['ordinal_entropy'] = -(probabilities * log_probabilities).sum(
            axis=1, min_count=6
        )
        mixture = (probabilities + 1.0 / 6.0) / 2.0
        out['mixture_entropy'] = -(mixture * np.log(mixture)).sum(axis=1, min_count=6)
        out['ordinal_divergence'] = (
            out['mixture_entropy'] - 0.5 * out['ordinal_entropy'] - 0.5 * np.log(6.0)
        ).clip(lower=0.0)
        out['complexity'] = (
            out['ordinal_entropy'] / np.log(6.0)
            * out['ordinal_divergence'] / np.log(2.0)
        )
        out['complexity_gain'] = out['complexity'] - out['complexity'].shift(params.comparison_lag)
        out['descending_past'] = out['probability_7'].shift(params.comparison_lag)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = np.isfinite(indicators[[
            'drawdown', 'complexity_gain', 'descending_past', 'atr'
        ]]).all(axis=1)
        ready &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        setup = (
            ready
            & indicators['drawdown'].between(params.min_drawdown, params.max_drawdown)
            & indicators['complexity_gain'].ge(params.complexity_rise)
            & indicators['descending_past'].ge(params.descending_threshold)
            & indicators['atr'].gt(0.0)
            & data['close'].gt(data['close'].shift(1))
        ).fillna(False)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy(dtype=bool)

        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if active:
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                else:
                    # The first tracked close is the next bar, when entry fills.
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    held_bars = i - entry_bar + 1
                    if close[i] <= stop or held_bars >= params.max_hold_bars:
                        active = False
            elif confirmed[i]:
                active = True
                entry_bar = i + 1
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)

        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = StrategyParams()
    strategy = GeneratedStrategy()
    warmup = strategy.warmup_bars(params)
    closes = [100.0] * warmup + [101.0, 102.0, 104.0, 105.0, 104.0, 102.0, 101.0]
    index = pd.date_range('2020-01-01', periods=len(closes), freq='B')
    data = pd.DataFrame({
        'open': closes, 'high': np.array(closes) + 1.0,
        'low': np.array(closes) - 1.0, 'close': closes, 'volume': 1000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['complexity'].dropna().between(0.0, 1.0).all()
    indicators = pd.DataFrame({
        'drawdown': 0.05, 'complexity_gain': 0.05,
        'descending_past': 0.6, 'atr': 1.0,
    }, index=index)
    indicators.iloc[warmup + 4:, indicators.columns.get_loc('atr')] = 10.0
    result = strategy.generate_signals(data, indicators, None, params).data
    # Two confirmations, next-bar entry, and no loosening after ATR expands.
    assert result['signal'].iloc[warmup:].tolist() == [0, 0, 1, 1, 1, 1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
