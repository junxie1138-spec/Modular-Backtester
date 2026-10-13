from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRhythmParams:
    entropy_window: int = 18
    entropy_ceiling: float = 0.75
    atr_window: int = 14
    minimum_gap_atr: float = 0.25
    stop_atr: float = 1.5
    holding_days: int = 4


class GeneratedStrategy(BaseStrategy[GapRhythmParams]):
    strategy_id = 'gen_local-w2_1790967961'

    @classmethod
    def params_type(cls) -> type[GapRhythmParams]:
        return GapRhythmParams

    def warmup_bars(self, params: GapRhythmParams) -> int:
        return int(max(params.entropy_window + 4, params.atr_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapRhythmParams) -> pd.DataFrame:
        if params.entropy_window < 6 or params.atr_window < 2:
            raise ValueError('Lookback windows are too short.')
        if not 0.0 <= params.entropy_ceiling <= 1.0:
            raise ValueError('Entropy ceiling must be between zero and one.')
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('Stop multiple must be positive and finite.')
        if not np.isfinite(params.minimum_gap_atr) or params.minimum_gap_atr <= 0:
            raise ValueError('Minimum gap must be positive and finite.')
        if params.holding_days not in (3, 4, 5):
            raise ValueError('Holding horizon must be three to five sessions.')

        out = pd.DataFrame(index=data.index)
        previous = data['close'].shift(1)
        out['previous_close'] = previous
        out['gap'] = (data['open'] / previous.where(previous > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['previous_atr'] = out['atr'].shift(1)

        a, b, c = out['gap'].shift(2), out['gap'].shift(1), out['gap']
        valid = a.notna() & b.notna() & c.notna()
        # Equal gaps are ordered chronologically, making ties deterministic.
        out['pattern'] = ((a <= b).astype(int) * 4 + (a <= c).astype(int) * 2 + (b <= c).astype(int)).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            probability = out['pattern'].eq(code).astype(float).where(valid).rolling(
                params.entropy_window, min_periods=params.entropy_window
            ).mean()
            out[f'pattern_probability_{code}'] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
        out['entropy'] = entropy / np.log(6.0)
        out['prior_entropy'] = out['entropy'].shift(1)
        out['ordered_gaps'] = out['prior_entropy'].le(params.entropy_ceiling).fillna(False)
        out['gap_reclaim'] = (
            (out['previous_atr'] > 0)
            & ((previous - data['open']) >= params.minimum_gap_atr * out['previous_atr'])
            & (data['close'] > previous)
        ).fillna(False)
        out['entry'] = (
            out['ordered_gaps'] & out['gap_reclaim']
            & out['atr'].gt(0) & np.isfinite(out['atr'])
            & np.isfinite(data['close'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapRhythmParams) -> SignalFrame:
        count = len(data)
        decisions = np.zeros(count, dtype=int)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        frozen_atr = np.nan
        stop_price = np.nan

        for i in range(count):
            if active:
                if i == entry_bar:
                    # Anchor to the execution bar's open only once that bar arrives.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    stop_price = opens[i] - params.stop_atr * frozen_atr
                expired = i - entry_bar + 1 >= params.holding_days
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                if expired or stopped:
                    active = False
                    continue
                decisions[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry_bar = i + 1
                frozen_atr = atr[i]
                decisions[i] = 1

        df = pd.DataFrame({'signal': decisions, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 96.0,
                         'close': [100.0, 101.0, 97.0, 100.0, 100.0, 100.0, 100.0],
                         'volume': 1000.0}, index=index)
    derived = pd.DataFrame({'entry': [True, False, False, False, False, False, False],
                            'atr': [2.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]}, index=index)
    strategy = GeneratedStrategy()
    params = GapRhythmParams(stop_atr=1.0)
    stopped = strategy.generate_signals(data, derived, None, params).data
    assert stopped['signal'].tolist() == [0, 1, 1, 0, 0, 0, 0]
    data['close'] = 100.0
    expired = strategy.generate_signals(data, derived, None, params).data
    assert expired['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert expired['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
