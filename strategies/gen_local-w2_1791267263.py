from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SellerClearanceParams:
    seller_window: int = 40
    volume_window: int = 20
    volume_threshold: float = 1.3
    seller_band: float = 0.5
    atr_window: int = 14
    atr_multiple: float = 3.0
    holding_bars: int = 18

    def __post_init__(self):
        if min(self.seller_window, self.volume_window, self.atr_window, self.holding_bars) < 1:
            raise ValueError('Lookbacks and holding_bars must be positive')
        if not all(np.isfinite(x) for x in (self.volume_threshold, self.seller_band, self.atr_multiple)):
            raise ValueError('Thresholds must be finite')
        if self.volume_threshold <= 0 or self.atr_multiple <= 0 or self.seller_band < 0:
            raise ValueError('Invalid threshold')


class GeneratedStrategy(BaseStrategy[SellerClearanceParams]):
    strategy_id = 'gen_local-w2_1791267263'

    @classmethod
    def params_type(cls):
        return SellerClearanceParams

    def warmup_bars(self, params: SellerClearanceParams) -> int:
        return int(max(200, params.seller_window + 1, params.volume_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: SellerClearanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        volume = data['volume'].clip(lower=0)
        out['previous_close'] = close.shift(1)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        out['down_volume'] = volume.where(close < out['previous_close'], 0.0)
        out['down_volume'] = out['down_volume'].where(out['previous_close'].notna())
        out['seller_volume'] = out['down_volume'].rolling(params.seller_window, min_periods=params.seller_window).sum()
        out['seller_value'] = (out['down_volume'] * close).rolling(params.seller_window, min_periods=params.seller_window).sum()
        out['seller_squared_value'] = (out['down_volume'] * close.pow(2)).rolling(params.seller_window, min_periods=params.seller_window).sum()
        denominator = out['seller_volume'].replace(0.0, np.nan)
        out['seller_mean'] = out['seller_value'] / denominator
        out['seller_std'] = (out['seller_squared_value'] / denominator - out['seller_mean'].pow(2)).clip(lower=0.0).pow(0.5)
        out['seller_position'] = (close - out['seller_mean']) / out['seller_std'].replace(0.0, np.nan)
        out['volume_baseline'] = volume.shift(1).rolling(params.volume_window, min_periods=params.volume_window).mean()
        out['relative_volume'] = volume / out['volume_baseline'].replace(0.0, np.nan)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['regime'] = (close > out['ma200']).fillna(False)
        out['entry'] = (
            out['regime']
            & (out['seller_position'] > params.seller_band)
            & (out['seller_position'].shift(1) <= params.seller_band)
            & (out['relative_volume'] >= params.volume_threshold)
            & (close > out['previous_close'])
            & (close > data['open'])
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SellerClearanceParams) -> SignalFrame:
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        regime = indicators['regime'].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = 0
        highest_close = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0
            if active:
                if not valid:
                    active = False
                else:
                    highest_close = max(highest_close, close[i])
                    stop = max(stop, highest_close - params.atr_multiple * atr[i])
                    age = i - entry_bar + 1
                    if close[i] <= stop or not regime[i] or age >= params.holding_bars:
                        active = False
            elif valid and entries[i] and regime[i]:
                active = True
                entry_bar = i + 1
                highest_close = -np.inf
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [200.0, 100.0, 104.0, 102.0, 101.0, 103.0]}, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0, 1.0, 1.0, 10.0, 10.0, 1.0],
        'entry': [True, False, False, False, False, False],
        'regime': True,
    }, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, indicators, None, SellerClearanceParams())
    assert result.data['signal'].tolist() == [0, 1, 1, 1, 1, 0]
    result = strategy.generate_signals(data, indicators, None, SellerClearanceParams(holding_bars=2))
    assert result.data['signal'].tolist() == [0, 1, 1, 0, 0, 0]
    assert strategy.warmup_bars(SellerClearanceParams(seller_window=250)) == 251
    assert (result.data['size'] > 0).all()


if __name__ == '__main__':
    _self_check()
