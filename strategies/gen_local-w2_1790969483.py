from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionEscapeParams:
    overlap_floor: float = 0.55
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[AuctionEscapeParams]):
    strategy_id = 'gen_local-w2_1790969483'

    @classmethod
    def params_type(cls):
        return AuctionEscapeParams

    def warmup_bars(self, params):
        if not np.isfinite(params.overlap_floor) or not 0 < params.overlap_floor < 1:
            raise ValueError('overlap_floor must lie between zero and one')
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError('trail_atr must be positive and finite')
        return 15

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        high = data['high']
        low = data['low']
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['bar_range'] = (high - low).where(high >= low)
        out['intersection'] = (np.minimum(high, high.shift(1)) - np.maximum(low, low.shift(1))).clip(lower=0)
        out['union'] = np.maximum(high, high.shift(1)) - np.minimum(low, low.shift(1))
        out['overlap'] = out['intersection'].div(out['union'].where(out['union'] > 0))
        out['prior_overlap'] = out['overlap'].rolling(4, min_periods=4).mean().shift(1)
        out['upper'] = high.rolling(6, min_periods=6).max().shift(1)
        out['lower'] = low.rolling(6, min_periods=6).min().shift(1)
        out['baseline_range'] = out['bar_range'].rolling(6, min_periods=6).median().shift(1)
        out['high_gap'] = (high - out['previous_close']).abs()
        out['low_gap'] = (low - out['previous_close']).abs()
        out['true_range'] = out[['bar_range', 'high_gap', 'low_gap']].max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(14, min_periods=14).mean()
        valid = np.isfinite(out[['prior_overlap', 'upper', 'lower', 'baseline_range', 'atr', 'bar_range']]).all(axis=1)
        gate = valid & (out['atr'] > 0) & (out['baseline_range'] > 0)
        gate &= (out['prior_overlap'] >= params.overlap_floor)
        gate &= (out['bar_range'] >= 1.3 * out['baseline_range'])
        out['entry'] = np.select([gate & (close > out['upper']), gate & (close < out['lower'])], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        self.warmup_bars(params)
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        side = 0
        pending = 0
        age = 0
        entry_price = 0.0
        stop = -np.inf
        peak = -np.inf
        armed = False
        for i in range(len(data)):
            if pending:
                # A decision from the preceding close enters at this open.
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                    raise ValueError('Cannot initialize entry from invalid open or ATR')
                side = pending
                pending = 0
                entry_price = opens[i]
                peak = side * entry_price
                # Signed coordinates make the stop ratchet upward for both sides.
                stop = peak - params.trail_atr * atr[i - 1]
                age = 0
                armed = False
            if side:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    continue
                signed_close = side * closes[i]
                # Stops are close-confirmed; flattening executes next bar.
                if signed_close <= stop or age >= 10:
                    side = 0
                    continue
                peak = max(peak, signed_close)
                if signed_close - side * entry_price >= 0.01 * entry_price:
                    armed = True
                if armed:
                    stop = max(stop, side * entry_price, peak - params.trail_atr * atr[i])
                decisions[i] = side
            elif entries[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                pending = int(entries[i])
                decisions[i] = pending
        df['signal'] = decisions
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = AuctionEscapeParams()
    index = pd.date_range('2020-01-01', periods=5)
    for side in (1, -1):
        close = 100.0 + side * np.array([0.0, 1.2, 2.0, 0.9, 0.5])
        data = pd.DataFrame({'open': 100.0, 'high': np.maximum(close, 100.0) + 0.1, 'low': np.minimum(close, 100.0) - 0.1, 'close': close, 'volume': 1000.0}, index=index)
        indicators = pd.DataFrame({'atr': 0.5, 'entry': [side, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, side, side, side, 0]
        assert (result['size'] > 0).all()
        warm = strategy.indicators(data, params)
        assert (warm['entry'] == 0).all()


if __name__ == '__main__':
    _self_check()
