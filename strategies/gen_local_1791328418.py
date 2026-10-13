from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    slope_window: int = 8
    compression_window: int = 3
    atr_window: int = 20
    compression_ratio: float = 0.85
    innovation_threshold: float = 0.025
    stop_atr: float = 1.25
    hold_bars: int = 2

    def __post_init__(self):
        if self.slope_window < 2 or self.compression_window < 1 or self.atr_window < 2:
            raise ValueError('Invalid indicator windows')
        if self.compression_window > self.atr_window:
            raise ValueError('Compression window must not exceed ATR window')
        values = (self.compression_ratio, self.innovation_threshold, self.stop_atr)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError('Thresholds must be finite and positive')
        if self.compression_ratio >= 1 or self.hold_bars not in (1, 2):
            raise ValueError('Require compression below one and a one- or two-session hold')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791328418'

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.slope_window, params.compression_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['compressed_range'] = out['true_range'].rolling(params.compression_window).mean()
        positive_atr = out['atr'].where(out['atr'] > 0)
        out['compression'] = out['compressed_range'] / positive_atr
        out['typical_price'] = (data['high'] + data['low'] + data['close']) / 3.0
        clock = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        n = params.slope_window
        out['latent_slope'] = out['typical_price'].rolling(n).cov(clock, ddof=0) / ((n * n - 1) / 12.0)
        out['slope_innovation'] = out['latent_slope'].diff() / positive_atr
        valid = np.isfinite(out['compression']) & np.isfinite(out['slope_innovation'])
        compressed = valid & (out['compression'] <= params.compression_ratio)
        out['candidate'] = np.select([
            compressed & (out['slope_innovation'] >= params.innovation_threshold),
            compressed & (out['slope_innovation'] <= -params.innovation_threshold),
        ], [1, -1], default=0).astype(int)
        out['confirmed'] = out['candidate'].where(
            out['candidate'].eq(out['candidate'].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        atr = indicators['atr'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        pending = False
        stop = np.nan
        age = 0

        for i in range(len(data)):
            if position != 0:
                if pending:
                    # A decision from the previous close fills at this open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        pending = False
                        continue
                    stop = opens[i] - position * params.stop_atr * atr[i - 1]
                    age = 0
                    pending = False
                age += 1
                stopped = np.isfinite(closes[i]) and (
                    (position == 1 and closes[i] < stop) or
                    (position == -1 and closes[i] > stop)
                )
                if stopped or age >= params.hold_bars:
                    position = 0
            elif i > 0:
                confirmed = candidates[i] != 0 and candidates[i] == candidates[i - 1]
                if confirmed and np.isfinite(atr[i]) and atr[i] > 0:
                    position = int(candidates[i])
                    pending = True
            raw[i] = position

        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': 100.0, 'high': 104.0, 'low': 96.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    ind = pd.DataFrame({'candidate': [1, 1, 0, 0, 0, 0], 'atr': 2.0}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 0]
    data.loc[index[2], 'close'] = 97.0
    assert strategy.generate_signals(data, ind, None, params).data['signal'].tolist() == [0, 0, 1, 0, 0, 0]
    ind['candidate'] = [-1, -1, 0, 0, 0, 0]
    data.loc[index[2], 'close'] = 103.0
    assert strategy.generate_signals(data, ind, None, params).data['signal'].tolist() == [0, 0, -1, 0, 0, 0]
    ind['candidate'] = [1, 0, 0, 0, 0, 0]
    assert strategy.generate_signals(data, ind, None, params).data['signal'].eq(0).all()
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
