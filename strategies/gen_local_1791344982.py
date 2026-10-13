from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    cluster_window: int = 5
    rank_window: int = 126
    atr_window: int = 14
    entry_percentile: float = 0.80
    trail_atr: float = 2.0
    max_hold: int = 10

    def __post_init__(self):
        if min(self.cluster_window, self.rank_window, self.atr_window, self.max_hold) < 1:
            raise ValueError('Windows and holding period must be positive.')
        if not 0.0 < self.entry_percentile < 1.0:
            raise ValueError('entry_percentile must lie between zero and one.')
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0.0:
            raise ValueError('trail_atr must be finite and positive.')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791344982'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.cluster_window + params.rank_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        previous = close.shift(1)
        out['return'] = (close / previous.where(previous > 0.0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out['downside'] = (-out['return']).clip(lower=0.0)
        out['adjacent_loss'] = out['downside'] * out['downside'].shift(1)
        out['loss_cluster'] = out['adjacent_loss'].rolling(params.cluster_window, min_periods=params.cluster_window).sum()
        out['cluster_percentile'] = out['loss_cluster'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous).abs(),
            (data['low'] - previous).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['daily_range'] = data['high'] - data['low']
        out['close_location'] = ((close - data['low']) / out['daily_range'].where(out['daily_range'] > 0.0)).clip(0.0, 1.0)
        out['severity'] = ((out['cluster_percentile'] - params.entry_percentile) / (1.0 - params.entry_percentile)).clip(0.0, 1.0)
        out['recovery'] = ((out['close_location'] - 0.5) * 2.0).clip(0.0, 1.0)
        out['conviction'] = (out['severity'] * out['recovery']).fillna(0.0)
        out['entry'] = (
            (out['cluster_percentile'] >= params.entry_percentile)
            & (out['loss_cluster'] > 0.0)
            & (out['return'] > 0.0)
            & (out['return'].shift(1) < 0.0)
            & (out['close_location'] >= 0.65)
            & (out['atr'] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        conviction = indicators['conviction'].fillna(0.0).to_numpy(dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        exposure = 1.0

        for i in range(count):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                age += 1
                if valid:
                    # Entry fills on this bar after yesterday's decision.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close <= stop or age >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = exposure
            elif valid and entries[i] and np.isfinite(conviction[i]):
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                exposure = float(0.25 + 0.70 * np.clip(conviction[i], 0.0, 1.0))
                signals[i] = 1
                sizes[i] = exposure

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame({'close': [99.0, 100.0, 101.0, 97.0, 97.0]}, index=index)
    indicators = pd.DataFrame({
        'atr': [1.0, 1.0, 10.0, 10.0, 1.0],
        'entry': [True, False, False, False, False],
        'conviction': [0.8, 0.0, 0.0, 0.0, 0.0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, Params()).data
    # ATR expansion must not lower the stop established at 98.
    assert result['signal'].tolist() == [0, 1, 1, 1, 0]
    assert np.allclose(result.loc[index[1:4], 'size'], 0.81)
    assert (result['size'] > 0.0).all()
    timed = strategy.generate_signals(data, indicators, None, Params(max_hold=2)).data
    assert timed['signal'].tolist() == [0, 1, 1, 0, 0]


if __name__ == '__main__':
    _self_check()
