from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    overlap_bars: int = 4
    fast_atr_bars: int = 5
    slow_atr_bars: int = 30
    compression_ratio: float = 0.8
    profit_target: float = 0.012
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791262001'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.fast_atr_bars + 2,
                       params.slow_atr_bars + 2, params.overlap_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.overlap_bars, params.fast_atr_bars,
               params.slow_atr_bars, params.holding_bars) < 1:
            raise ValueError('Lookbacks and holding_bars must be positive.')
        if not (np.isfinite(params.compression_ratio)
                and params.compression_ratio > 0
                and np.isfinite(params.profit_target)
                and params.profit_target > 0):
            raise ValueError('Compression ratio and profit target must be positive and finite.')
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['fast_atr'] = out['true_range'].rolling(
            params.fast_atr_bars, min_periods=params.fast_atr_bars).mean()
        out['slow_atr'] = out['true_range'].rolling(
            params.slow_atr_bars, min_periods=params.slow_atr_bars).mean()
        out['prior_compression'] = (
            out['fast_atr'] / out['slow_atr'].replace(0.0, np.nan)
        ).shift(1)
        out['intersection_top'] = data['high'].rolling(
            params.overlap_bars, min_periods=params.overlap_bars).min().shift(1)
        out['intersection_bottom'] = data['low'].rolling(
            params.overlap_bars, min_periods=params.overlap_bars).max().shift(1)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        ready = np.isfinite(out).all(axis=1)
        out['entry'] = (
            ready
            & (data['close'] > out['ma200'])
            & (out['prior_compression'] <= params.compression_ratio)
            & (out['intersection_top'] > out['intersection_bottom'])
            & (out['previous_close'] <= out['intersection_top'])
            & (data['close'] > out['intersection_top'])
            & (data['close'] > data['open'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.holding_bars
                if invalid_entry or target_hit or expired:
                    # Close-observed exits execute at the following bar's fill.
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif entries[i]:
                entry_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = Params()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': 100.0, 'high': 103.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({'entry': [True] + [False] * 7}, index=index)
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[1], 'close'] = 102.0
    result = strategy.generate_signals(data, indicators, None, params)
    assert result.data['signal'].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert not strategy.indicators(data, params)['entry'].any()


if __name__ == '__main__':
    _self_check()
