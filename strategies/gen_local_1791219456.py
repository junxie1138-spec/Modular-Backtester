from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapGraphParams:
    graph_lookback: int = 20
    percentile_window: int = 126
    percentile: float = 0.75
    atr_length: int = 14
    breakeven_pct: float = 0.01
    trail_atr: float = 1.8
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GapGraphParams]):
    strategy_id = 'gen_local_1791219456'

    @classmethod
    def params_type(cls):
        return GapGraphParams

    def warmup_bars(self, params: GapGraphParams) -> int:
        return int(params.atr_length + params.graph_lookback + params.percentile_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapGraphParams) -> pd.DataFrame:
        if not (params.graph_lookback >= 3 and params.percentile_window >= 2
                and params.atr_length >= 2 and 0.0 < params.percentile < 1.0
                and params.breakeven_pct > 0.0 and params.trail_atr > 0.0
                and 3 <= params.hold_days <= 5):
            raise ValueError('Invalid gap graph parameters')
        out = pd.DataFrame(index=data.index)
        previous_close = data['close'].shift(1)
        out['previous_close'] = previous_close
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_length).mean()
        out['previous_atr'] = out['atr'].shift(1).where(lambda s: s > 0.0)
        out['gap'] = ((data['open'] - previous_close) / out['previous_atr']).replace([np.inf, -np.inf], np.nan)
        out['magnitude'] = out['gap'].abs()
        magnitude = out['magnitude']
        blocker = pd.Series(-np.inf, index=data.index)
        degree = pd.Series(0.0, index=data.index)
        votes = pd.Series(0.0, index=data.index)
        # Strict horizontal visibility: intervening magnitudes lie below both endpoints.
        for lag in range(1, params.graph_lookback + 1):
            neighbor = magnitude.shift(lag)
            visible = ((magnitude > blocker) & (neighbor > blocker)
                       & magnitude.notna() & neighbor.notna())
            degree += visible.astype(float)
            votes += visible.astype(float) * np.sign(out['gap'].shift(lag)).fillna(0.0)
            blocker = np.maximum(blocker, neighbor)
        complete = magnitude.rolling(params.graph_lookback + 1).count().eq(params.graph_lookback + 1)
        out['blocker'] = blocker
        out['degree'] = degree.where(complete)
        out['neighbor_bias'] = (votes / degree.replace(0.0, np.nan)).where(complete)
        out['dissent'] = -np.sign(out['gap']) * out['neighbor_bias'] * out['degree']
        out['dissent_threshold'] = out['dissent'].shift(1).rolling(params.percentile_window).quantile(params.percentile)
        out['gap_threshold'] = magnitude.shift(1).rolling(params.percentile_window).quantile(params.percentile)
        out['intraday_move'] = data['close'] - data['open']
        valid = (np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
                 & data[['open', 'high', 'low', 'close']].gt(0.0).all(axis=1)
                 & out['atr'].gt(0.0) & out['dissent_threshold'].notna()
                 & out['gap_threshold'].notna())
        enter = (valid & out['dissent'].gt(out['dissent_threshold'])
                 & magnitude.gt(out['gap_threshold'])
                 & (np.sign(out['gap']) * out['neighbor_bias']).lt(0.0)
                 & (np.sign(out['gap']) * out['intraday_move']).lt(0.0))
        out['entry'] = np.where(enter, -np.sign(out['gap']).fillna(0.0), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapGraphParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['entry'].fillna(0).to_numpy(dtype=int)
        side = 0
        entry_bar = -1
        entry_price = 0.0
        stop = -np.inf
        best = -np.inf
        armed = False
        for i in range(count):
            if side:
                if i == entry_bar:
                    entry_price = opens[i]
                    entry_atr = atr[i - 1]
                    if not (np.isfinite(entry_price) and entry_price > 0.0
                            and np.isfinite(entry_atr) and entry_atr > 0.0):
                        side = 0
                        continue
                    best = side * entry_price
                    stop = best - params.trail_atr * entry_atr
                    armed = False
                if not (np.isfinite(highs[i]) and np.isfinite(lows[i])
                        and np.isfinite(atr[i]) and atr[i] > 0.0):
                    side = 0
                    continue
                adverse = lows[i] if side == 1 else -highs[i]
                # Stops established before this bar are tested before tightening.
                # Exit decisions execute at the following open after the mandatory shift.
                if adverse <= stop or i - entry_bar + 1 >= params.hold_days:
                    side = 0
                    continue
                favorable = highs[i] if side == 1 else -lows[i]
                best = max(best, favorable)
                if best - side * entry_price >= params.breakeven_pct * entry_price:
                    armed = True
                if armed:
                    # The stop only rises in direction-adjusted coordinates;
                    # therefore a short position's actual price stop only falls.
                    stop = max(stop, side * entry_price, best - params.trail_atr * atr[i])
                signals[i] = side
            elif candidates[i] != 0:
                side = int(candidates[i])
                entry_bar = i + 1
                signals[i] = side
        df = pd.DataFrame({'signal': signals, 'size': np.ones(count, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({
        'open': [100, 101, 102, 100, 100, 100, 98, 100],
        'high': [101, 103, 103, 101, 101, 100, 101, 101],
        'low': [99, 101, 100, 99, 99, 97, 97, 99],
        'close': [100, 102, 101, 100, 100, 98, 100, 100],
        'volume': [1000] * 8,
    }, index=index)
    indicators = pd.DataFrame({'atr': [1.0] * 8, 'entry': [1, 0, 0, 0, -1, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, GapGraphParams()).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 0, -1, -1, 0]
    assert result['size'].gt(0.0).all()
    assert result.index.equals(data.index)


if __name__ == '__main__':
    _self_check()
