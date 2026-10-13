from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapChangeParams:
    recent_bars: int = 4
    baseline_bars: int = 40
    change_threshold: float = 1.8
    positive_fraction: float = 0.75
    atr_bars: int = 14
    initial_stop_atr: float = 2.5
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    holding_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapChangeParams]):
    strategy_id = 'gen_local_1791222342'

    @classmethod
    def params_type(cls) -> type[GapChangeParams]:
        return GapChangeParams

    @staticmethod
    def warmup_bars(params: GapChangeParams) -> int:
        return int(max(200, params.recent_bars + params.baseline_bars + 1, params.atr_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapChangeParams) -> pd.DataFrame:
        if params.recent_bars < 2 or params.baseline_bars < 2 or params.atr_bars < 1:
            raise ValueError('Lookback windows must be positive; gap windows require at least two bars.')
        if not (np.isfinite(params.change_threshold) and params.change_threshold > 0):
            raise ValueError('change_threshold must be positive and finite.')
        if not 0 < params.positive_fraction <= 1:
            raise ValueError('positive_fraction must be in (0, 1].')
        if params.holding_bars < 1:
            raise ValueError('holding_bars must be positive.')
        if any(not np.isfinite(x) or x <= 0 for x in (params.initial_stop_atr, params.breakeven_pct, params.trail_atr)):
            raise ValueError('Stop parameters must be positive and finite.')
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1).where(lambda x: x > 0)
        out['gap'] = (data['open'] / previous_close - 1).replace([np.inf, -np.inf], np.nan)
        historical_gap = out['gap'].shift(params.recent_bars)
        out['recent_mean'] = out['gap'].rolling(params.recent_bars, min_periods=params.recent_bars).mean()
        out['baseline_mean'] = historical_gap.rolling(params.baseline_bars, min_periods=params.baseline_bars).mean()
        out['baseline_std'] = historical_gap.rolling(params.baseline_bars, min_periods=params.baseline_bars).std(ddof=1)
        out['standard_error'] = out['baseline_std'].clip(lower=0.0001) * np.sqrt(1.0 / params.recent_bars + 1.0 / params.baseline_bars)
        out['change_score'] = (out['recent_mean'] - out['baseline_mean']) / out['standard_error']
        positive_gap = (out['gap'] > 0).astype(float).where(out['gap'].notna())
        out['positive_fraction'] = positive_gap.rolling(params.recent_bars, min_periods=params.recent_bars).mean()
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        out['true_range'] = pd.concat([(data['high'] - data['low']), (data['high'] - previous_close).abs(), (data['low'] - previous_close).abs()], axis=1).max(axis=1).where(previous_close.notna())
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['ready'] = np.isfinite(out[['change_score', 'recent_mean', 'positive_fraction', 'ma200', 'atr']]).all(axis=1) & out['atr'].gt(0)
        out['condition'] = (out['ready'] & out['change_score'].ge(params.change_threshold) & out['recent_mean'].gt(0) & out['positive_fraction'].ge(params.positive_fraction) & data['close'].gt(out['ma200'])).fillna(False)
        out['entry_event'] = out['condition'] & ~out['condition'].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapChangeParams) -> SignalFrame:
        df = pd.DataFrame({'signal': np.zeros(len(data), dtype=int), 'size': np.ones(len(data), dtype=float)}, index=data.index)
        prices = data[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        ready = indicators['ready'].fillna(False).to_numpy(dtype=bool)
        events = indicators['entry_event'].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = 0.0
        peak = 0.0
        stop = 0.0
        age = 0
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            valid = ready[i] and np.isfinite(prices[i]).all() and low > 0 and low <= min(opening, close) and high >= max(opening, close)
            if not valid:
                active = False
                pending = False
                continue
            if pending:
                # The preceding close scheduled this entry at today's open.
                entry = opening
                peak = opening
                stop = max(0.0, entry - params.initial_stop_atr * atr[i - 1])
                age = 0
                armed = False
                active = True
                pending = False
            if active:
                age += 1
                # Stop crossings are observed at the close and exit next open.
                # Test the previously established stop before updating it.
                if low <= stop or close <= ma[i] or age >= params.holding_bars:
                    active = False
                    continue
                peak = max(peak, high)
                armed = armed or high >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                raw[i] = 1
            elif events[i]:
                pending = True
                raw[i] = 1
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'open': [100, 102, 103, 104, 103, 103], 'high': [101, 103, 105, 104, 104, 104], 'low': [99, 101, 102, 102.5, 102, 102], 'close': [100, 102, 104, 103, 103, 103], 'volume': [1000] * 6}, index=index)
    indicators = pd.DataFrame({'atr': [1.0] * 6, 'ma200': [90.0] * 6, 'ready': [True] * 6, 'entry_event': [True, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, GapChangeParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 0, 0]
    expiry = GeneratedStrategy.generate_signals(data, indicators, None, GapChangeParams(holding_bars=2))
    assert expiry.data['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    assert result.data['size'].gt(0).all()
    assert GeneratedStrategy.warmup_bars(GapChangeParams()) >= 200


if __name__ == '__main__':
    _self_check()
