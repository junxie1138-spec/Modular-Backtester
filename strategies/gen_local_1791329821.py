from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    atr_period: int = 14
    flow_period: int = 3
    breakout_period: int = 20
    gap_atr: float = 0.35
    stop_atr: float = 1.5
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.atr_period, self.flow_period, self.breakout_period) < 1:
            raise ValueError('Lookback periods must be positive')
        if not np.isfinite(self.gap_atr) or self.gap_atr <= 0:
            raise ValueError('gap_atr must be finite and positive')
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError('stop_atr must be finite and positive')
        if not 3 <= self.hold_bars <= 5:
            raise ValueError('hold_bars must be between three and five')


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = 'gen_local_1791329821'

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.atr_period + 2,
                       params.flow_period + params.breakout_period))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['previous_high'] = data['high'].shift(1)
        out['previous_low'] = data['low'].shift(1)
        out['range'] = data['high'] - data['low']
        out['true_range'] = pd.concat([
            out['range'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_period).mean()
        out['reference_atr'] = out['atr'].shift(1)
        out['gap'] = (data['open'] - out['previous_close']) / out['reference_atr'].replace(0, np.nan)
        out['intraday_direction'] = (
            (data['close'] - data['open']) / out['range'].replace(0, np.nan)
        ).clip(-1.0, 1.0).fillna(0.0)
        out['signed_volume'] = out['intraday_direction'] * data['volume']
        out['volume_sum'] = data['volume'].rolling(params.flow_period).sum()
        out['signed_volume_sum'] = out['signed_volume'].rolling(params.flow_period).sum()
        out['flow'] = out['signed_volume_sum'] / out['volume_sum'].replace(0, np.nan)
        out['flow_upper'] = out['flow'].shift(1).rolling(params.breakout_period).max()
        out['flow_lower'] = out['flow'].shift(1).rolling(params.breakout_period).min()
        ready = np.isfinite(out[['gap', 'atr', 'flow', 'flow_upper', 'flow_lower']]).all(axis=1)
        ready &= out['atr'].gt(0)
        long_entry = (
            ready
            & out['gap'].ge(params.gap_atr)
            & data['open'].gt(out['previous_high'])
            & out['flow'].gt(out['flow_upper'])
            & out['flow'].gt(0)
        )
        short_entry = (
            ready
            & out['gap'].le(-params.gap_atr)
            & data['open'].lt(out['previous_low'])
            & out['flow'].lt(out['flow_lower'])
            & out['flow'].lt(0)
        )
        out['entry'] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].to_numpy(dtype=int)
        raw = np.zeros(len(data), dtype=int)
        side = 0
        fill_bar = -1
        entry_atr = np.nan
        stop = np.nan

        for i in range(len(data)):
            if side:
                if i == fill_bar:
                    # Capture the actual fill-bar open only when that bar arrives.
                    stop = opens[i] - side * params.stop_atr * entry_atr
                held = i - fill_bar + 1
                stopped = side * (closes[i] - stop) < 0
                invalid = not (np.isfinite(stop) and np.isfinite(closes[i]))
                if invalid or stopped or held >= params.hold_bars:
                    side = 0
                    continue
                raw[i] = side
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(entries[i])
                fill_bar = i + 1
                # Freeze ATR at the decision close; never trail the stop.
                entry_atr = atr[i]
                raw[i] = side

        df = pd.DataFrame(index=data.index)
        df['signal'] = raw
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range('2020-01-01', periods=8)
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=0)
    for side in (1, -1):
        data = pd.DataFrame({
            'open': 100.0,
            'high': 110.0,
            'low': 90.0,
            'close': [100.0, 100.0 + side * 4, 100.0 + side * 5,
                      100.0 - side, 100.0, 100.0, 100.0, 100.0],
            'volume': 1000.0,
        }, index=index)
        computed = strategy.indicators(data, params)
        assert computed.index.equals(data.index)
        assert computed['entry'].eq(0).all()
        indicators = pd.DataFrame({
            'atr': 2.0,
            'entry': [side, 0, 0, 0, 0, 0, 0, 0],
        }, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        # Four sessions, including a pullback that must not trigger a trailing stop.
        assert result['signal'].tolist() == [0, side, side, side, side, 0, 0, 0]
        assert result['size'].gt(0).all()
        # A fill-price gap changes the fixed stop; decision-close pricing is wrong.
        data.loc[index[1], 'open'] = 100.0 + side * 10
        data.loc[index[1], 'close'] = 100.0 + side * 6
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result['signal'].tolist() == [0, side, 0, 0, 0, 0, 0, 0]


if __name__ == '__main__':
    _self_check()
