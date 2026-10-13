from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RestorationParams:
    trend_bars: int = 12
    anchor_bars: int = 30
    rank_bars: int = 126
    atr_bars: int = 14
    tail_quantile: float = 0.8
    trail_atr: float = 2.0
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[RestorationParams]):
    strategy_id = 'gen_local_1791253986'

    @classmethod
    def params_type(cls) -> type[RestorationParams]:
        return RestorationParams

    def warmup_bars(self, params: RestorationParams) -> int:
        return int(max(params.trend_bars + 1, params.anchor_bars, params.atr_bars) + params.rank_bars + 1)

    def indicators(self, data: pd.DataFrame, params: RestorationParams) -> pd.DataFrame:
        if min(params.trend_bars, params.anchor_bars, params.rank_bars, params.atr_bars) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.5 < params.tail_quantile < 1.0 or params.trail_atr <= 0 or params.max_hold < 1:
            raise ValueError('Invalid percentile, trailing distance, or holding period')
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['log_close'] = np.log(close)
        out['bar_position'] = np.arange(len(data), dtype=float)
        out['slope'] = (
            out['log_close'].rolling(params.trend_bars).cov(out['bar_position'])
            / out['bar_position'].rolling(params.trend_bars).var()
        )
        out['innovation'] = out['slope'].diff()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean().where(lambda x: x > 0)
        out['atr_fraction'] = out['atr'] / close
        out['peak'] = close.rolling(params.anchor_bars).max()
        out['trough'] = close.rolling(params.anchor_bars).min()
        out['drawdown'] = np.log(out['peak'] / close)
        out['runup'] = np.log(close / out['trough'])
        out['down_displacement'] = pd.concat([out['drawdown'], out['atr_fraction']], axis=1).max(axis=1, skipna=False)
        out['up_displacement'] = pd.concat([out['runup'], out['atr_fraction']], axis=1).max(axis=1, skipna=False)
        out['long_restoration'] = out['innovation'] / out['down_displacement']
        out['short_restoration'] = -out['innovation'] / out['up_displacement']
        for column in ('drawdown', 'runup', 'long_restoration', 'short_restoration'):
            out[column + '_threshold'] = out[column].rolling(params.rank_bars).quantile(params.tail_quantile).shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['entry_long'] = (
            (out['drawdown'] > out['drawdown_threshold'])
            & (out['slope'] < 0)
            & (out['long_restoration'] > 0)
            & (out['long_restoration'] > out['long_restoration_threshold'])
        ).fillna(False)
        out['entry_short'] = (
            (out['runup'] > out['runup_threshold'])
            & (out['slope'] > 0)
            & (out['short_restoration'] > 0)
            & (out['short_restoration'] > out['short_restoration_threshold'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RestorationParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        long_entry = indicators['entry_long'].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators['entry_short'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = age = 0
        watermark = stop = np.nan
        for i, price in enumerate(closes):
            if not np.isfinite(price) or price <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = age = 0
                watermark = stop = np.nan
                continue
            if position:
                age += 1
                # The entry decision close precedes the fill; start the watermark on the next bar.
                if position == 1:
                    watermark = price if np.isnan(watermark) else max(watermark, price)
                    candidate = watermark - params.trail_atr * atr[i]
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    stopped = price <= stop
                else:
                    watermark = price if np.isnan(watermark) else min(watermark, price)
                    candidate = watermark + params.trail_atr * atr[i]
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    stopped = price >= stop
                if stopped or age >= params.max_hold:
                    position = 0
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                age = 0
                watermark = stop = np.nan
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=7)
    expected = np.array([0, 1, 1, 1, 1, 0, 0])
    for side, prices in ((1, [100, 102, 104, 103, 101, 100, 99]), (-1, [100, 98, 96, 97, 99, 100, 101])):
        data = pd.DataFrame({'close': prices}, index=index)
        indicators = pd.DataFrame({
            'atr': [1, 1, 1, 10, 1, 1, 1],
            'entry_long': [side == 1] + [False] * 6,
            'entry_short': [side == -1] + [False] * 6,
        }, index=index)
        result = strategy.generate_signals(data, indicators, None, RestorationParams()).data
        assert np.array_equal(result['signal'].to_numpy(), side * expected)
        assert (result['size'] > 0).all()
        result = strategy.generate_signals(data, indicators, None, RestorationParams(max_hold=2)).data
        assert np.array_equal(result['signal'].to_numpy(), side * np.array([0, 1, 1, 0, 0, 0, 0]))


if __name__ == '__main__':
    _self_check()
