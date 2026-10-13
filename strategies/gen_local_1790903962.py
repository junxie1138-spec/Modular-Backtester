from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    signature_bars: int = 12
    atr_bars: int = 14
    area_threshold: float = 0.20
    stop_atr: float = 2.5
    holding_bars: int = 18

    def __post_init__(self):
        if self.signature_bars < 2 or self.atr_bars < 1 or self.holding_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive; signature_bars must exceed one.')
        if not np.isfinite(self.area_threshold) or not 0 <= self.area_threshold <= 1:
            raise ValueError('area_threshold must be between zero and one.')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive.')


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = 'gen_local_1790903962'

    @classmethod
    def params_type(cls):
        return SignatureSeasonParams

    def warmup_bars(self, params):
        return int(max(200, params.signature_bars + 1, params.atr_bars + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        length = params.signature_bars
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['return_path'] = out['return'].fillna(0.0).cumsum()
        out['net_return'] = out['return'].rolling(length, min_periods=length).sum()
        out['absolute_return'] = out['return'].abs().rolling(length, min_periods=length).sum()
        # Twice the signed time-price area: sum((2*j-L-1) * return_j).
        out['prefix_sum'] = out['return_path'].shift(1).rolling(length, min_periods=length).sum() - length * out['return_path'].shift(length)
        out['area'] = ((length - 1) * out['net_return'] - 2.0 * out['prefix_sum']) / ((length - 1) * out['absolute_return'].replace(0.0, np.nan))
        out['area'] = out['area'].replace([np.inf, -np.inf], np.nan)
        out['ma200'] = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['month_start'] = data.index.day <= 7
        out['month_key'] = data.index.year * 12 + data.index.month
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        close = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        ma = indicators['ma200'].to_numpy(dtype=float)
        entry = (
            indicators['month_start']
            & indicators['area'].ge(params.area_threshold)
            & indicators['net_return'].gt(0.0)
            & data['close'].gt(indicators['ma200'])
            & indicators['atr'].gt(0.0)
        ).fillna(False).to_numpy(dtype=bool)
        months = indicators['month_key'].to_numpy(dtype=int)
        active = False
        entered_at = -1
        last_entry_month = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(ma[i])
            if active:
                if not valid:
                    active = False
                else:
                    # Entry fills on the next bar; start tracking its closing high here.
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.stop_atr * atr[i])
                    if close[i] <= stop or close[i] <= ma[i] or i - entered_at >= params.holding_bars:
                        active = False
            elif valid and entry[i] and months[i] != last_entry_month:
                active = True
                entered_at = i
                last_entry_month = months[i]
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df['signal'] = signals
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = SignatureSeasonParams(stop_atr=2.0)
    index = pd.bdate_range('2024-01-02', periods=5)
    close = np.array([100.0, 104.0, 101.0, 105.0, 106.0])
    data = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1, 'close': close, 'volume': 1000.0}, index=index)
    features = pd.DataFrame({'atr': [1.0, 1.0, 10.0, 1.0, 1.0], 'ma200': 90.0, 'month_start': True, 'area': 0.8, 'net_return': 0.01, 'month_key': 2024 * 12 + 1}, index=index)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=200)
    result = strategy.generate_signals(data, features, ctx, params).data
    # Expanded ATR cannot lower the existing stop; reentry waits for a new month.
    assert result['signal'].tolist() == [0, 1, 1, 0, 0]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(params) >= 200
    prices = np.array([100.0, 100.0, 100.0, 101.0])
    sample = pd.DataFrame({'open': prices, 'high': prices + 1, 'low': prices - 1, 'close': prices, 'volume': 1000.0}, index=index[:4])
    area = strategy.indicators(sample, SignatureSeasonParams(signature_bars=3))['area'].iloc[-1]
    assert np.isclose(area, 1.0)


if __name__ == '__main__':
    _self_check()
