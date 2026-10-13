from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangepointParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    recent_window: int = 4
    baseline_window: int = 20
    jump_threshold: float = 1.25
    atr_window: int = 14
    stop_atr: float = 2.0
    holding_days: int = 4

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError('fractional_order must be between zero and one')
        if min(self.fractional_window, self.recent_window, self.baseline_window, self.atr_window) < 2:
            raise ValueError('indicator windows must be at least two')
        if not 3 <= self.holding_days <= 5:
            raise ValueError('holding_days must be three to five')
        if not np.isfinite(self.jump_threshold) or self.jump_threshold <= 0:
            raise ValueError('jump_threshold must be finite and positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')


class GeneratedStrategy(BaseStrategy[ChangepointParams]):
    strategy_id = 'gen_local-w2_1791324918'

    @classmethod
    def params_type(cls) -> type[ChangepointParams]:
        return ChangepointParams

    @staticmethod
    def warmup_bars(params: ChangepointParams) -> int:
        return int(max(200, params.fractional_window + params.recent_window + params.baseline_window - 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ChangepointParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        weights = np.ones(params.fractional_window, dtype=float)
        for lag in range(1, params.fractional_window):
            weights[lag] = -weights[lag - 1] * (params.fractional_order - lag + 1) / lag
        fractional = np.full(len(data), np.nan, dtype=float)
        # ponytail: finite fractional kernel; increase fractional_window if omitted history matters.
        if len(data) >= params.fractional_window:
            fractional[params.fractional_window - 1:] = np.convolve(close.to_numpy(), weights, mode='valid')
        out['fractional_close'] = fractional
        out['recent_mean'] = out['fractional_close'].rolling(params.recent_window).mean()
        baseline = out['fractional_close'].shift(params.recent_window)
        out['baseline_mean'] = baseline.rolling(params.baseline_window).mean()
        out['baseline_std'] = baseline.rolling(params.baseline_window).std(ddof=0)
        out['mean_jump'] = ((out['recent_mean'] - out['baseline_mean']) / out['baseline_std'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['previous_jump'] = out['mean_jump'].shift(1)
        out['ma200'] = close.rolling(200).mean()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['entry'] = (
            out['mean_jump'].gt(params.jump_threshold)
            & out['previous_jump'].le(params.jump_threshold)
            & out['fractional_close'].gt(out['recent_mean'])
            & close.gt(out['ma200'])
            & out['atr'].gt(0.0)
            & np.isfinite(out['atr'])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ChangepointParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        entry_atr = 0.0
        stop_price = np.nan
        age = 0
        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if pending:
                pending = False
                active = bool(np.isfinite(opens[i]) and opens[i] > 0.0)
                if active:
                    stop_price = opens[i] - params.stop_atr * entry_atr
                    age = 0
            if active:
                age += 1
                # The stop remains fixed at entry; exits fill next bar.
                if not np.isfinite(closes[i]) or closes[i] < stop_price or age >= params.holding_days:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                pending = True
                entry_atr = atr[i]
                decisions[i] = 1
        df['signal'] = decisions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'atr': 1.0, 'entry': [True, False, False, False, False, False, False]}, index=index)
    params = ChangepointParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[1], 'open'] = 105.0
    data.loc[index[1], 'close'] = 102.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, 0, 0, 0]
    assert result['size'].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200


if __name__ == '__main__':
    _self_check()
