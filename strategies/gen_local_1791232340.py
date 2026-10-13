from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    training_window: int = 126
    horizon: int = 5
    entry_threshold: float = 0.12
    complexity_floor: float = 0.04
    min_samples: int = 8
    shrinkage: float = 8.0
    min_size: float = 0.15
    max_size: float = 0.95

    def __post_init__(self):
        if self.training_window < 6 or self.horizon < 1:
            raise ValueError('Invalid lookback')
        if not 1 <= self.min_samples <= self.training_window:
            raise ValueError('Invalid sample requirement')
        if not np.isfinite(self.shrinkage) or self.shrinkage < 0:
            raise ValueError('Invalid shrinkage')
        if not np.isfinite(self.entry_threshold) or self.entry_threshold <= 0:
            raise ValueError('Invalid entry threshold')
        if not 0 < self.complexity_floor < 1:
            raise ValueError('Invalid complexity floor')
        if not 0 < self.min_size <= self.max_size <= 1:
            raise ValueError('Invalid sizing bounds')


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local_1791232340'

    @classmethod
    def params_type(cls) -> type[OrdinalParams]:
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(params.training_window + 4 * params.horizon)

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        h, w = params.horizon, params.training_window
        close = data['close'].astype(float).replace(0.0, np.nan)
        out['weekly_return'] = close.pct_change(h, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['previous_return'] = out['weekly_return'].shift(h)
        out['oldest_return'] = out['weekly_return'].shift(2 * h)
        a, b, c = out['oldest_return'], out['previous_return'], out['weekly_return']
        valid = a.notna() & b.notna() & c.notna()
        conditions = [
            (a <= b) & (b <= c),
            (a <= c) & (c < b),
            (b < a) & (a <= c),
            (b <= c) & (c < a),
            (c < a) & (a <= b),
            (c < b) & (b < a),
        ]
        out['pattern'] = pd.Series(np.select(conditions, list(range(6)), default=np.nan), index=data.index).where(valid)
        # The pattern h bars ago is paired with its now-observed h-bar outcome.
        out['training_pattern'] = out['pattern'].shift(h)
        out['outcome_scale'] = c.rolling(w, min_periods=w).std().replace(0.0, np.nan)
        entropy = pd.Series(0.0, index=data.index)
        disequilibrium = pd.Series(0.0, index=data.index)
        edge = pd.Series(np.nan, index=data.index)
        training_valid = out['training_pattern'].notna() & c.notna()
        for k in range(6):
            probability = out['pattern'].eq(k).astype(float).where(valid).rolling(w, min_periods=w).mean()
            out[f'probability_{k}'] = probability
            entropy = entropy - probability * np.log(probability.replace(0.0, 1.0)) / np.log(6.0)
            disequilibrium = disequilibrium + (probability - 1.0 / 6.0) ** 2
            matches = out['training_pattern'].eq(k)
            count = matches.astype(float).where(training_valid).rolling(w, min_periods=w).sum()
            total = c.where(matches, 0.0).where(training_valid).rolling(w, min_periods=w).sum()
            out[f'samples_{k}'] = count
            out[f'outcome_sum_{k}'] = total
            out[f'forecast_{k}'] = (total / (count + params.shrinkage)).where(count >= params.min_samples)
            conditional_edge = out[f'forecast_{k}'] / out['outcome_scale']
            edge = edge.where(~out['pattern'].eq(k), conditional_edge)
        out['entropy'] = entropy
        out['disequilibrium'] = disequilibrium / (5.0 / 6.0)
        out['complexity'] = (entropy * np.sqrt(out['disequilibrium'].clip(lower=0.0))).clip(0.0, 1.0)
        out['edge'] = edge.replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        edge = indicators['edge'].replace([np.inf, -np.inf], np.nan)
        complexity = indicators['complexity'].replace([np.inf, -np.inf], np.nan)
        qualified = edge.notna() & complexity.ge(params.complexity_floor)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[qualified & edge.ge(params.entry_threshold)] = 1
        candidate.loc[qualified & edge.le(-params.entry_threshold)] = -1
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(candidate.to_numpy()):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Flatten on an opposite qualified entry; re-entry requires a later bar.
                position = 0
            positions[i] = position
        conviction = (edge.abs() / (3.0 * params.entry_threshold)).clip(0.0, 1.0)
        structure = (complexity / (3.0 * params.complexity_floor)).clip(0.0, 1.0)
        decision_size = params.min_size + (params.max_size - params.min_size) * conviction * structure
        df = pd.DataFrame({'signal': positions}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = decision_size.shift(1).fillna(params.min_size).clip(params.min_size, params.max_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = OrdinalParams()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({'edge': [1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0], 'complexity': 0.2}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result['size'].between(params.min_size, params.max_size).all()
    assert result['size'].iloc[1] > result['size'].iloc[2]
