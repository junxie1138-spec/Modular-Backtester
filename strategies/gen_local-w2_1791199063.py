from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 40
    percentile_window: int = 126
    depth_quantile: float = 0.85
    atr_window: int = 14
    stop_atr: float = 2.0
    max_hold: int = 8
    size: float = 0.95

    def __post_init__(self):
        if min(self.peak_window, self.percentile_window, self.atr_window, self.max_hold) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.0 < self.depth_quantile < 1.0:
            raise ValueError('depth_quantile must be between zero and one.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0.0:
            raise ValueError('stop_atr must be finite and positive.')
        if not np.isfinite(self.size) or not 0.0 < self.size <= 1.0:
            raise ValueError('size must be between zero and one.')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local-w2_1791199063'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.peak_window + params.percentile_window + 1, params.atr_window + 1, 3))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['previous_low'] = data['low'].shift(1)
        out['peak'] = data['close'].shift(1).rolling(params.peak_window).max()
        out['drawdown'] = (1.0 - data['close'] / out['peak'].where(out['peak'] > 0.0)).clip(lower=0.0)
        out['depth_threshold'] = out['drawdown'].shift(1).rolling(params.percentile_window).quantile(params.depth_quantile)
        out['gap'] = data['open'] / out['previous_close'].where(out['previous_close'] > 0.0) - 1.0
        out['previous_gap'] = out['gap'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out['entry'] = (
            ready
            & (out['depth_threshold'] > 0.0)
            & (out['drawdown'] >= out['depth_threshold'])
            & (out['previous_gap'] < out['gap'])
            & (out['gap'] < 0.0)
            & (data['low'] >= out['previous_low'])
            & (data['close'] < out['previous_close'])
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GeneratedParams) -> SignalFrame:
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        stop = None
        width = 0.0
        filled_bar = 0

        for i in range(len(data)):
            if active:
                if stop is None:
                    # Entry fills at this open; ATR was frozen at the preceding decision close.
                    stop = opens[i] - width
                    filled_bar = i
                active = bool(
                    np.isfinite(stop)
                    and np.isfinite(closes[i])
                    and closes[i] >= stop
                    and i - filled_bar + 1 < params.max_hold
                )
                if not active:
                    stop = None
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                active = True
                stop = None
                width = params.stop_atr * atr[i]
            raw[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = float(params.size)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=5, freq='B')
    data = pd.DataFrame({
        'open': [90.0, 100.0, 110.0, 97.0, 95.0],
        'high': [91.0, 111.0, 111.0, 98.0, 96.0],
        'low': [89.0, 99.0, 96.0, 94.0, 94.0],
        'close': [90.0, 110.0, 97.0, 95.0, 95.0],
        'volume': [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({'entry': [True, False, False, False, False], 'atr': [2.0] * 5}, index=index)
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    # Fixed stop is 100 - 2 * 2 = 96, independent of the subsequent rally.
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    params.max_hold = 2
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    assert (result['size'] > 0.0).all()
    assert not strategy.indicators(data, params)['entry'].any()


if __name__ == '__main__':
    _self_check()
