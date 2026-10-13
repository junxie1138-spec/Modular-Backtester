from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    pattern_window: int = 24
    vol_window: int = 20
    low_entropy: float = 0.65
    high_entropy: float = 0.90
    min_move: float = 0.50
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = 'gen_local_1791278277'

    @classmethod
    def params_type(cls) -> type[OrdinalParams]:
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(max(params.pattern_window + 3, params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        if params.pattern_window < 2 or params.vol_window < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0.0 < params.low_entropy < params.high_entropy < 1.0:
            raise ValueError('Entropy thresholds must satisfy 0 < low < high < 1.')
        if params.min_move <= 0 or params.holding_bars not in (1, 2):
            raise ValueError('Minimum move must be positive; holding bars must be one or two.')

        out = pd.DataFrame(index=data.index)
        returns = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        triples = np.column_stack((returns.shift(2), returns.shift(1), returns))
        valid = np.isfinite(triples).all(axis=1)
        # Stable ordering breaks equal-return ties chronologically.
        order = np.argsort(triples, axis=1, kind='stable')
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        out['return'] = returns
        out['previous_return'] = returns.shift(1)
        out['ordinal_code'] = pd.Series(codes, index=data.index).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (5, 7, 11, 15, 19, 21):
            occurrence = out['ordinal_code'].eq(code).astype(float).where(valid)
            probability = occurrence.rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            out[f'pattern_probability_{code}'] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out['entropy'] = (entropy / np.log(6.0)).clip(0.0, 1.0)
        out['volatility'] = returns.rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0)
        out['move'] = returns.abs() / out['volatility'].replace(0.0, np.nan)
        span = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['close_location'] = ((data['close'] - data['low']) / span).clip(0.0, 1.0)

        orderly = out['entropy'].le(params.low_entropy)
        disordered = out['entropy'].ge(params.high_entropy)
        continuation = orderly & returns.gt(0.0) & returns.gt(out['previous_return'])
        absorption = disordered & returns.lt(0.0)
        ready = np.isfinite(out[['entropy', 'move', 'close_location']]).all(axis=1)
        out['entry'] = ready & (continuation | absorption) & out['move'].ge(params.min_move) & out['close_location'].ge(0.70)
        orderly_confidence = ((params.low_entropy - out['entropy']) / params.low_entropy).clip(0.0, 1.0)
        disorder_confidence = ((out['entropy'] - params.high_entropy) / (1.0 - params.high_entropy)).clip(0.0, 1.0)
        out['regime_confidence'] = pd.Series(np.where(orderly, orderly_confidence, disorder_confidence), index=data.index)
        out['strength'] = (out['move'].clip(0.0, 3.0) / 3.0 * (0.5 + 0.5 * out['regime_confidence'])).fillna(0.0).clip(0.0, 1.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalParams) -> SignalFrame:
        if params.holding_bars not in (1, 2):
            raise ValueError('Holding bars must be one or two.')
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        strength = indicators['strength'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.10, dtype=float)
        exit_at = -1
        entry_size = 0.10
        for i in range(len(data)):
            if exit_at >= 0:
                if i == exit_at:
                    exit_at = -1
                else:
                    signals[i] = 1
                    sizes[i] = entry_size
            elif entries[i]:
                exit_at = i + params.holding_bars
                entry_size = float(0.10 + 0.85 * strength[i])
                signals[i] = 1
                sizes[i] = entry_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.10).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    evidence = pd.DataFrame({'entry': True, 'strength': np.linspace(0.0, 1.0, 7)}, index=index)
    strategy = GeneratedStrategy()
    for holding, expected in ((1, [0, 1, 0, 1, 0, 1, 0]), (2, [0, 1, 1, 0, 1, 1, 0])):
        result = strategy.generate_signals(data, evidence, None, OrdinalParams(holding_bars=holding)).data
        assert result['signal'].tolist() == expected
        assert result['size'].between(0.10, 0.95).all()
        if holding == 2:
            assert result['size'].iloc[1] == result['size'].iloc[2]
            assert result['size'].iloc[4] > result['size'].iloc[1]


if __name__ == '__main__':
    _self_check()
