from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_bars: int = 12
    volume_bars: int = 20
    peak_bars: int = 63
    percentile_bars: int = 126
    tail_percentile: float = 0.20
    depth_percentile: float = 0.65
    confirmation_bars: int = 3
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local-w2_1791245888'

    @classmethod
    def params_type(cls):
        return SignatureParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.peak_bars, params.volume_bars + params.path_bars + 1, params.confirmation_bars + 1) + params.percentile_bars + 1)

    def indicators(self, data, params):
        if min(params.path_bars, params.volume_bars, params.peak_bars, params.percentile_bars, params.confirmation_bars, params.max_hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if params.path_bars < 2 or not 0.0 < params.tail_percentile < 0.5 or not 0.0 < params.depth_percentile < 1.0 or not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError('Invalid signature, percentile, or target parameters.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['price_increment'] = np.log(close).diff()
        out['mean_volume'] = data['volume'].where(data['volume'] >= 0).rolling(params.volume_bars).mean()
        out['relative_volume'] = data['volume'].where(data['volume'] >= 0).div(out['mean_volume'].where(out['mean_volume'] > 0))
        spread = (data['high'] - data['low']).where(data['high'] >= data['low'])
        location = (2.0 * close - data['high'] - data['low']).div(spread.where(spread > 0)).clip(-1.0, 1.0)
        out['close_location'] = location.mask(spread.eq(0), 0.0)
        out['pressure_increment'] = out['close_location'] * out['relative_volume']
        x = out['price_increment']
        y = out['pressure_increment']
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, params.path_bars):
            cross = x.shift(lag) * y - y.shift(lag) * x
            area = area + 0.5 * cross.rolling(params.path_bars - lag).sum()
        out['signature_area'] = area
        out['area_scale'] = x.abs().rolling(params.path_bars).sum() * y.abs().rolling(params.path_bars).sum()
        out['normalized_area'] = area.div(out['area_scale'].where(out['area_scale'] > 0))
        out['area_lower'] = out['normalized_area'].shift(1).rolling(params.percentile_bars).quantile(params.tail_percentile)
        out['area_upper'] = out['normalized_area'].shift(1).rolling(params.percentile_bars).quantile(1.0 - params.tail_percentile)
        out['rolling_peak'] = close.rolling(params.peak_bars).max()
        out['drawdown_depth'] = 1.0 - close.div(out['rolling_peak'])
        out['depth_threshold'] = out['drawdown_depth'].shift(1).rolling(params.percentile_bars).quantile(params.depth_percentile)
        out['confirmation_return'] = close.div(close.shift(params.confirmation_bars)) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1)
        deep = (out['drawdown_depth'] > out['depth_threshold']) & (out['drawdown_depth'] > 0.0)
        long_entry = valid & deep & (out['normalized_area'] < out['area_lower']) & (out['confirmation_return'] > 0.0)
        short_entry = valid & deep & (out['normalized_area'] > out['area_upper']) & (out['confirmation_return'] < 0.0)
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        if params.max_hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError('Holding period and profit target must be positive.')
        raw = np.zeros(len(data), dtype=np.int64)
        close = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        side, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if side and i == entry_bar:
                entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0.0:
                    side = 0
                    continue
            if side:
                # ponytail: targets use the entry open before execution costs; use fill records if the contract exposes them.
                gain = side * (close[i] / entry_price - 1.0)
                if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold_bars:
                    side = 0
                raw[i] = side
            elif entries[i] and i + 1 < len(data):
                side = int(entries[i])
                entry_bar = i + 1
                raw[i] = side
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = object.__new__(GeneratedStrategy)
    params = SignatureParams(max_hold_bars=2, profit_target=0.03)
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame({'open': 100.0, 'high': 106.0, 'low': 94.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'entry': [1, 0, 0, -1, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    data.loc[index[1], 'close'] = 105.0
    data.loc[index[4], 'close'] = 95.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 0, -1, 0, 0]
    assert result['size'].gt(0).all()
    assert strategy.indicators(data, params)['entry'].eq(0).all()


if __name__ == '__main__':
    _self_check()
