from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 12
    breakout_window: int = 20
    atr_window: int = 14
    area_threshold: float = 0.08
    stop_atr: float = 2.5
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.path_window, self.breakout_window, self.atr_window) < 2:
            raise ValueError('Lookback windows must be at least two.')
        if not 0.0 < self.area_threshold < 0.5:
            raise ValueError('Area threshold must lie between zero and 0.5.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0 or self.hold_bars < 1:
            raise ValueError('Stop multiplier and holding period must be positive.')
        if not 0.0 < self.min_size <= self.max_size <= 1.0:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1790921405'

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        return max(params.path_window + 1, params.breakout_window + 1, params.atr_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        length = params.path_window
        area = pd.Series(0.0, index=data.index)
        # For increments (1/L, return_j), antisymmetric level-two area is
        # sum_j ((j - (L-1)/2)/L) * return_j, ordered oldest to newest.
        for j in range(length):
            area = area + ((j - (length - 1) / 2.0) / length) * returns.shift(length - 1 - j)
        variation = returns.abs().rolling(length, min_periods=length).sum()
        out['signature_area'] = area.div(variation.where(variation > 0))
        out['path_return'] = returns.rolling(length, min_periods=length).sum()
        out['resistance'] = data['high'].rolling(params.breakout_window, min_periods=params.breakout_window).max().shift(1)
        previous = close.shift(1)
        true_range = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        area_strength = (out['signature_area'] / (3.0 * params.area_threshold)).clip(0.0, 1.0)
        out['entry_size'] = params.min_size + (params.max_size - params.min_size) * area_strength
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        count = len(data)
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        strength = indicators['entry_size'].to_numpy(dtype=float)
        ready = np.isfinite(indicators[['signature_area', 'path_return', 'resistance', 'atr', 'entry_size']]).all(axis=1)
        candidate = (
            ready
            & (data['close'] > indicators['resistance'])
            & (indicators['signature_area'] >= params.area_threshold)
            & (indicators['path_return'] > 0)
            & (indicators['atr'] > 0)
            & (data['close'] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(count, dtype=int)
        size = np.ones(count, dtype=float)
        active = False
        pending = False
        entry_index = 0
        entry_atr = 0.0
        entry_size = 1.0
        stop = 0.0
        for i in range(count):
            if pending:
                pending = False
                active = True
                entry_index = i
                # Entry occurs at this opening; ATR is frozen at the prior decision.
                stop = opens[i] - params.stop_atr * entry_atr
            if active:
                if not np.isfinite(close[i]) or not np.isfinite(stop) or close[i] < stop or i - entry_index + 1 >= params.hold_bars:
                    active = False
                else:
                    signal[i] = 1
                    size[i] = entry_size
            elif candidate[i] and i + 1 < count:
                pending = True
                entry_atr = atr[i]
                entry_size = strength[i]
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({'signal': signal, 'size': size}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=7, freq='D')
    data = pd.DataFrame({
        'open': [100., 101., 102., 110., 105., 94., 100.],
        'high': [101., 102., 111., 111., 106., 101., 101.],
        'low': [99., 100., 101., 105., 93., 93., 99.],
        'close': [100., 101., 110., 106., 94., 100., 100.],
        'volume': [1000.] * 7,
    }, index=index)
    params = SignatureParams()
    derived = GeneratedStrategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert derived['signature_area'].isna().all()
    indicators = pd.DataFrame({
        'signature_area': [0.12] * 7,
        'path_return': [0.1] * 7,
        'resistance': [99.] * 7,
        'atr': [2.] * 7,
        'entry_size': [0.6] * 7,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 1]
    assert (result['size'] > 0).all()
    assert np.isclose(result['size'].iloc[1], 0.6)
    expired = GeneratedStrategy.generate_signals(data, indicators, None, SignatureParams(hold_bars=2)).data
    assert expired['signal'].iloc[3] == 0
    indicators.loc[:, 'signature_area'] = np.nan
    flat = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert (flat['signal'] == 0).all()


if __name__ == '__main__':
    _self_check()
