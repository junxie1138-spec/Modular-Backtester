from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionParams:
    range_bars: int = 20
    percentile_bars: int = 126
    percentile: float = 0.85
    atr_bars: int = 14
    stop_atr: float = 2.0
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[AuctionParams]):
    strategy_id = 'gen_local_1791200850'

    @classmethod
    def params_type(cls):
        return AuctionParams

    def warmup_bars(self, params: AuctionParams) -> int:
        return int(max(params.range_bars, params.atr_bars) + params.percentile_bars + 2)

    def indicators(self, data: pd.DataFrame, params: AuctionParams) -> pd.DataFrame:
        if min(params.range_bars, params.percentile_bars, params.atr_bars, params.holding_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not 0.5 < params.percentile < 1.0 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError('Invalid percentile or stop multiplier.')
        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1)
        out['prior_high'] = data['high'].shift(1)
        out['prior_low'] = data['low'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['prior_close']).abs(),
            (data['low'] - out['prior_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out['prior_atr'] = out['atr'].shift(1).replace(0.0, np.nan)
        out['range_high'] = data['high'].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out['range_low'] = data['low'].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        out['drawdown'] = (out['range_high'] - out['prior_close']) / out['range_high'].replace(0.0, np.nan)
        out['runup'] = (out['prior_close'] - out['range_low']) / out['range_low'].replace(0.0, np.nan)
        out['gap'] = (data['open'] - out['prior_close']) / out['prior_atr']
        out['absolute_gap'] = out['gap'].abs()
        for source, target in [('absolute_gap', 'gap_threshold'), ('drawdown', 'drawdown_threshold'), ('runup', 'runup_threshold')]:
            out[target] = out[source].rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.percentile).shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1) & (out['atr'] > 0)
        out['long_entry'] = (
            ready
            & (out['drawdown'] > out['drawdown_threshold'])
            & (out['gap'] < -out['gap_threshold'])
            & (data['open'] < out['range_low'])
            & (data['close'] > out['prior_high'])
        ).fillna(False)
        out['short_entry'] = (
            ready
            & (out['runup'] > out['runup_threshold'])
            & (out['gap'] > out['gap_threshold'])
            & (data['open'] > out['range_high'])
            & (data['close'] < out['prior_low'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AuctionParams) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['long_entry'].fillna(False).astype(int).to_numpy() - indicators['short_entry'].fillna(False).astype(int).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        side, entry_bar, stop = 0, -1, np.nan
        for i in range(len(data)):
            if side:
                if i == entry_bar:
                    # Anchor once at the next opening price using decision-bar ATR.
                    stop = opens[i] - side * params.stop_atr * atr[i - 1]
                if not np.isfinite(stop) or side * (closes[i] - stop) < 0 or i - entry_bar + 1 >= params.holding_bars:
                    side = 0
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidates[i])
                entry_bar = i + 1
            raw[i] = side
        df = pd.DataFrame({'signal': raw, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = AuctionParams()
    index = pd.date_range('2020-01-01', periods=11)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    for direction in (1, -1):
        opens = [100.0, 100.0 + direction * 10.0] + [100.0] * 9
        closes = [100.0, opens[1] - direction, opens[1] - direction * 3.0] + [100.0] * 8
        data = pd.DataFrame({'open': opens, 'close': closes}, index=index)
        ind = pd.DataFrame({'atr': 1.0, 'long_entry': False, 'short_entry': False}, index=index)
        ind.loc[index[0], 'long_entry' if direction == 1 else 'short_entry'] = True
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result['signal'].tolist() == [0, direction, direction] + [0] * 8
        assert (result['size'] > 0).all()
    data = pd.DataFrame({'open': 100.0, 'close': 100.0}, index=index)
    ind = pd.DataFrame({'atr': 1.0, 'long_entry': False, 'short_entry': False}, index=index)
    ind.loc[index[0], 'long_entry'] = True
    assert strategy.generate_signals(data, ind, ctx, params).data['signal'].tolist() == [0] + [1] * 8 + [0] * 2
