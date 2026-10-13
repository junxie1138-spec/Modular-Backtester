from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 63
    pattern_window: int = 20
    trough_window: int = 10
    min_drawdown: float = 0.02
    max_drawdown: float = 0.12
    min_repair: float = 0.15
    max_repair: float = 0.55
    complexity_floor: float = 0.10
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local-w2_1790967857'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        if min(params.peak_window, params.pattern_window, params.trough_window) < 3:
            raise ValueError('Lookback windows must be at least three bars')
        if params.hold_bars < 1:
            raise ValueError('hold_bars must be positive')
        if not 0 < params.min_drawdown < params.max_drawdown < 1:
            raise ValueError('Invalid drawdown bounds')
        if not 0 <= params.min_repair < params.max_repair <= 1:
            raise ValueError('Invalid recovery bounds')
        if not 0 < params.complexity_floor < 1:
            raise ValueError('Invalid complexity floor')
        return int(max(params.peak_window, params.trough_window, params.pattern_window + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['clean_close'] = close.where(close > 0)
        close = out['clean_close']
        out['lag_two'] = close.shift(2)
        out['lag_one'] = close.shift(1)
        a, b, c = out['lag_two'], out['lag_one'], close
        valid = a.notna() & b.notna() & c.notna()
        # Equal prices retain chronological order in the ordinal ranking.
        code = 4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)
        out['ordinal_code'] = code.astype(float).where(valid)
        probability_columns = []
        for motif in (0, 1, 3, 4, 6, 7):
            hit_column = f'motif_{motif}'
            probability_column = f'probability_{motif}'
            out[hit_column] = out['ordinal_code'].eq(motif).astype(float).where(valid)
            out[probability_column] = out[hit_column].rolling(params.pattern_window, min_periods=params.pattern_window).mean()
            probability_columns.append(probability_column)
        probabilities = out[probability_columns]
        out['entropy'] = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1, min_count=6) / np.log(6.0)
        out['disequilibrium'] = probabilities.sub(1.0 / 6.0).pow(2).sum(axis=1, min_count=6) / (5.0 / 6.0)
        out['complexity'] = out['entropy'] * out['disequilibrium']
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['trough'] = close.rolling(params.trough_window, min_periods=params.trough_window).min()
        out['drawdown'] = 1.0 - close / out['peak']
        depth = (out['peak'] - out['trough']).where(out['peak'] > out['trough'])
        out['repair'] = (close - out['trough']) / depth
        out['rising'] = close.gt(out['lag_one']) & close.notna() & out['lag_one'].notna()
        out['entry'] = (
            out['drawdown'].between(params.min_drawdown, params.max_drawdown)
            & out['repair'].between(params.min_repair, params.max_repair)
            & out['complexity'].ge(params.complexity_floor)
            & out['rising']
        ).fillna(False)
        out['entry_size'] = (0.25 + 0.70 * (out['complexity'] / 0.25).clip(0.0, 1.0)).fillna(0.25)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        entry_sizes = indicators['entry_size'].replace([np.inf, -np.inf], np.nan).fillna(0.25).clip(0.25, 0.95).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        exit_bar = -1
        held_size = 0.25
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = held_size
            elif entries[i]:
                exit_bar = i + params.hold_bars
                held_size = float(entry_sizes[i])
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = RecoveryParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=24, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert not derived['entry'].any()
    forced = pd.DataFrame({'entry': False, 'entry_size': 0.60}, index=index)
    forced.loc[index[0], 'entry'] = True
    forced.loc[index[params.hold_bars], 'entry'] = True
    result = strategy.generate_signals(data, forced, None, params).data
    assert result['signal'].tolist() == [0] + [1] * params.hold_bars + [0] * (len(data) - params.hold_bars - 1)
    assert result['size'].iloc[1:params.hold_bars + 1].eq(0.60).all()
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) >= params.pattern_window + 2


if __name__ == '__main__':
    _self_check()
