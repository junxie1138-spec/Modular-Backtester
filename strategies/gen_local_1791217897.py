from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 6
    volume_bars: int = 20
    percentile_bars: int = 126
    entry_percentile: float = 0.85
    atr_bars: int = 14
    trailing_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791217897'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(params.atr_bars + 1, params.volume_bars + params.path_bars + params.percentile_bars + 2))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        if min(params.path_bars, params.volume_bars, params.percentile_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.5 < params.entry_percentile < 1.0 or not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0:
            raise ValueError('Invalid percentile or trailing ATR multiplier.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        volume = data['volume'].where(data['volume'] >= 0)
        out['log_return'] = np.log(close).diff()
        out['volume_baseline'] = volume.shift(1).rolling(params.volume_bars, min_periods=params.volume_bars).mean()
        out['volume_surprise'] = volume.div(out['volume_baseline'].where(out['volume_baseline'] > 0)) - 1.0
        x = pd.Series(0.0, index=data.index)
        y = pd.Series(0.0, index=data.index)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-order signature: half the ordered cross-products.
        for lag in range(params.path_bars - 1, -1, -1):
            dx = out['log_return'].shift(lag)
            dy = out['volume_surprise'].shift(lag)
            area = area + x * dy - y * dx
            x = x + dx
            y = y + dy
        out['path_return'] = x
        out['path_volume_surprise'] = y
        out['return_variation'] = out['log_return'].abs().rolling(params.path_bars, min_periods=params.path_bars).sum()
        out['volume_variation'] = out['volume_surprise'].abs().rolling(params.path_bars, min_periods=params.path_bars).sum()
        out['area_scale'] = out['return_variation'] * out['volume_variation']
        out['signature_area'] = (0.5 * area).div(out['area_scale'].where(out['area_scale'] > 0))
        historical = out['signature_area'].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars)
        out['entry_threshold'] = historical.quantile(params.entry_percentile)
        out['reset_threshold'] = historical.quantile(0.5)
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry'] = (
            (out['signature_area'] > out['entry_threshold'])
            & (out['signature_area'] > 0)
            & (out['path_return'] < 0)
            & (out['log_return'] > 0)
            & (out['volume_surprise'] < 0)
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        active = False
        armed = True
        age = 0
        high_water = -np.inf
        stop = -np.inf
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        areas = indicators['signature_area'].to_numpy(dtype=float)
        resets = indicators['reset_threshold'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if active:
                # Entry fills on this bar when age is zero; begin tracking closes here.
                age += 1
                if np.isfinite(close) and close > 0:
                    high_water = max(high_water, close)
                    if np.isfinite(atr) and atr > 0:
                        stop = max(stop, high_water - params.trailing_atr * atr)
                    if close <= stop:
                        active = False
                else:
                    active = False
                if age >= params.hold_bars:
                    active = False
                signals[i] = int(active)
                continue
            if np.isfinite(areas[i]) and np.isfinite(resets[i]) and areas[i] <= resets[i]:
                armed = True
            if armed and entries[i] and np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0:
                active = True
                armed = False
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=12, freq='D')
    close = np.array([99, 100, 105, 104, 102, 102, 102, 102, 102, 102, 102, 102], dtype=float)
    data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 100.0}, index=index)
    indicators = pd.DataFrame({
        'atr': [1, 1, 1, 10, 10, 1, 1, 1, 1, 1, 1, 1],
        'signature_area': 0.4,
        'reset_threshold': 0.0,
        'entry': [True] + [False] * 11,
    }, index=index)
    strategy = GeneratedStrategy()
    params = SignatureParams(trailing_atr=2.0)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    indicators['atr'] = 1.0
    data.loc[:, 'close'] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0] + [1] * 8 + [0] * 3
    assert (result['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
