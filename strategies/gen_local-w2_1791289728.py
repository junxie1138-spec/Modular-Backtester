from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionRecoveryParams:
    auction_window: int = 12
    rank_window: int = 80
    extreme_window: int = 20
    tail_percentile: float = 0.8
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8

    def __post_init__(self):
        if min(self.auction_window, self.rank_window, self.extreme_window, self.atr_window) < 2:
            raise ValueError('Lookbacks must be at least two bars')
        if not 0.5 < self.tail_percentile < 1.0:
            raise ValueError('tail_percentile must lie between 0.5 and 1')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')
        if not 5 <= self.hold_bars <= 10:
            raise ValueError('hold_bars must be between five and ten sessions')


class GeneratedStrategy(BaseStrategy[AuctionRecoveryParams]):
    strategy_id = 'gen_local-w2_1791289728'

    @classmethod
    def params_type(cls):
        return AuctionRecoveryParams

    def warmup_bars(self, params):
        return int(max(params.auction_window + params.rank_window + 1,
                       params.extreme_window, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['change'] = close.diff()
        out['typical_price'] = (data['high'] + data['low'] + close) / 3.0
        out['usable_volume'] = data['volume'].clip(lower=0).where(out['change'].notna())
        out['down_volume'] = out['usable_volume'].where(out['change'] < 0, 0.0).where(out['change'].notna())
        out['up_volume'] = out['usable_volume'].where(out['change'] > 0, 0.0).where(out['change'].notna())
        out['down_notional'] = out['typical_price'] * out['down_volume']
        out['up_notional'] = out['typical_price'] * out['up_volume']
        w = params.auction_window
        out['total_turnover'] = out['usable_volume'].rolling(w, min_periods=w).sum()
        out['sell_turnover'] = out['down_volume'].rolling(w, min_periods=w).sum()
        out['buy_turnover'] = out['up_volume'].rolling(w, min_periods=w).sum()
        out['sell_notional'] = out['down_notional'].rolling(w, min_periods=w).sum()
        out['buy_notional'] = out['up_notional'].rolling(w, min_periods=w).sum()
        out['sell_price'] = out['sell_notional'] / out['sell_turnover'].replace(0, np.nan)
        out['buy_price'] = out['buy_notional'] / out['buy_turnover'].replace(0, np.nan)
        out['sell_share'] = out['sell_turnover'] / out['total_turnover'].replace(0, np.nan)
        out['buy_share'] = out['buy_turnover'] / out['total_turnover'].replace(0, np.nan)
        r = params.rank_window
        out['sell_percentile'] = out['sell_share'].rolling(r, min_periods=r).rank(pct=True)
        out['buy_percentile'] = out['buy_share'].rolling(r, min_periods=r).rank(pct=True)
        out['recent_peak'] = close.rolling(params.extreme_window, min_periods=params.extreme_window).max()
        out['recent_trough'] = close.rolling(params.extreme_window, min_periods=params.extreme_window).min()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out['long_candidate'] = (
            (out['sell_percentile'] >= params.tail_percentile)
            & (close < out['recent_peak'])
            & (close > out['sell_price'])
            & (out['change'] > 0)
            & (out['atr'] > 0)
        ).fillna(False)
        out['short_candidate'] = (
            (out['buy_percentile'] >= params.tail_percentile)
            & (close > out['recent_trough'])
            & (close < out['buy_price'])
            & (out['change'] < 0)
            & (out['atr'] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        long_candidate = indicators['long_candidate'].fillna(False).astype(bool)
        short_candidate = indicators['short_candidate'].fillna(False).astype(bool)
        long_entry = (long_candidate & long_candidate.shift(1, fill_value=False)).to_numpy()
        short_entry = (short_candidate & short_candidate.shift(1, fill_value=False)).to_numpy()
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        targets = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        stop = np.nan
        frozen_distance = np.nan
        for i in range(len(data)):
            if side:
                # Observe the entry open only when its bar has arrived.
                if i == entry_bar:
                    stop = opens[i] - side * frozen_distance
                valid = np.isfinite(closes[i]) and np.isfinite(stop)
                stopped = valid and (
                    (side == 1 and closes[i] < stop)
                    or (side == -1 and closes[i] > stop)
                )
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid or stopped or expired:
                    side = 0
            elif np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]):
                if long_entry[i] and not short_entry[i]:
                    side = 1
                elif short_entry[i] and not long_entry[i]:
                    side = -1
                if side:
                    entry_bar = i + 1
                    frozen_distance = params.stop_atr * atr[i]
                    stop = np.nan
            targets[i] = side
        df = pd.DataFrame({'signal': targets, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=10)
    data = pd.DataFrame({
        'open': 100.0,
        'high': 106.0,
        'low': 94.0,
        'close': [100, 100, 100, 99, 95, 100, 100, 101, 105, 100],
        'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        'long_candidate': [False, True, True, False, False, False, False, False, False, False],
        'short_candidate': [False, False, False, False, False, True, True, False, False, False],
        'atr': [np.nan, 2, 2, 100, 100, 2, 2, 100, 100, 100],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, AuctionRecoveryParams()).data
    assert result['signal'].tolist() == [0, 0, 0, 1, 1, 0, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    indicators['long_candidate'] = True
    indicators['short_candidate'] = False
    indicators['atr'] = 2.0
    data['close'] = 100.0
    result = strategy.generate_signals(data, indicators, None, AuctionRecoveryParams(hold_bars=5)).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]


if __name__ == '__main__':
    _self_check()
