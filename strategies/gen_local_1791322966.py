from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    motif_window: int = 30
    trend_window: int = 20
    complexity_floor: float = 0.04
    efficiency_floor: float = 0.25
    atr_window: int = 14
    trail_atr: float = 3.0
    max_holding: int = 20
    minimum_size: float = 0.20


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791322966'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.motif_window + 7, params.trend_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.motif_window, params.trend_window, params.atr_window, params.max_holding) < 1:
            raise ValueError('Lookbacks and maximum holding must be positive.')
        if not (0.0 <= params.complexity_floor <= 1.0 and 0.0 <= params.efficiency_floor < 1.0):
            raise ValueError('Invalid entry thresholds.')
        if not (params.trail_atr > 0.0 and 0.0 < params.minimum_size <= 1.0):
            raise ValueError('Invalid trailing distance or minimum size.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        a, b, c = out['return'].shift(2), out['return'].shift(1), out['return']
        # Equal values retain chronological order, giving deterministic ordinal ties.
        out['motif'] = ((a > b).astype(int) + 2 * (a > c).astype(int) + 4 * (b > c).astype(int)).where(a.notna() & b.notna() & c.notna())
        out['transition'] = 8 * out['motif'].shift(1) + out['motif']
        probability_columns = []
        motifs = (0, 1, 3, 4, 6, 7)
        for previous in motifs:
            for current in motifs:
                # The shared pair keeps its relative order: only 18 transitions are feasible.
                if bool(previous & 4) != bool(current & 1):
                    continue
                name = f'p_{previous}_{current}'
                out[name] = out['transition'].eq(8 * previous + current).astype(float).where(out['transition'].notna()).rolling(params.motif_window, min_periods=params.motif_window).mean()
                probability_columns.append(name)
        probabilities = out[probability_columns]
        out['entropy'] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1, min_count=18) / np.log(18.0)
        out['disequilibrium'] = ((probabilities.pow(2).sum(axis=1, min_count=18) - 1.0 / 18.0) / (1.0 - 1.0 / 18.0)).clip(0.0, 1.0)
        out['complexity'] = out['entropy'] * out['disequilibrium']
        out['complexity_growth'] = out['complexity'].diff(3)
        out['net_return'] = out['return'].rolling(params.trend_window, min_periods=params.trend_window).sum()
        out['absolute_return'] = out['return'].abs().rolling(params.trend_window, min_periods=params.trend_window).sum()
        out['efficiency'] = (out['net_return'] / out['absolute_return'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([data['high'] - data['low'], (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['strength'] = (out['efficiency'].clip(0.0, 1.0) * (out['complexity'] / 0.15).clip(0.0, 1.0)).fillna(0.0)
        out['ready'] = out[['complexity', 'complexity_growth', 'efficiency', 'atr']].notna().all(axis=1) & out['atr'].gt(0.0) & close.gt(0.0) & np.isfinite(close)
        out['ready'] &= np.arange(len(data)) >= self.warmup_bars(params) - 1
        out['entry'] = out['ready'] & out['complexity'].ge(params.complexity_floor) & out['complexity_growth'].gt(0.0) & out['efficiency'].ge(params.efficiency_floor)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.minimum_size, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strength = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        active = False
        entry_bar = 0
        high_water = 0.0
        stop = -np.inf
        for i in range(len(data)):
            if active:
                if not ready[i]:
                    active = False
                else:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if closes[i] <= stop or i - entry_bar >= params.max_holding:
                        active = False
            elif ready[i] and entries[i]:
                active = True
                entry_bar = i
                high_water = closes[i]
                stop = high_water - params.trail_atr * atr[i]
            if active:
                signals[i] = 1
                sizes[i] = params.minimum_size + (1.0 - params.minimum_size) * strength[i]
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params(trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [100.0, 102.0, 104.0, 101.0, 105.0]}, index=index)
    indicators = pd.DataFrame({'atr': [1.0, 1.0, 1.0, 10.0, 1.0], 'ready': True, 'entry': [True, False, False, False, False], 'strength': [0.2, 0.4, 0.8, 0.5, 0.1]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    assert np.allclose(result['size'].to_numpy(), [0.2, 0.36, 0.52, 0.84, 0.2])
    assert result['size'].gt(0.0).all()
    params.max_holding = 2
    assert strategy.generate_signals(data, indicators, None, params).data['signal'].tolist() == [0, 1, 1, 0, 0]


if __name__ == '__main__':
    _self_check()
