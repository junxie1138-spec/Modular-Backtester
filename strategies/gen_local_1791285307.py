from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    depth_window: int = 63
    gap_sum_window: int = 5
    gap_channel_window: int = 15
    long_depth: float = 0.04
    short_depth: float = 0.012
    atr_window: int = 14
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1791285307'

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.depth_window, params.gap_sum_window + params.gap_channel_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        out['previous_close'] = close.shift(1)
        out['peak'] = close.rolling(params.depth_window, min_periods=params.depth_window).max()
        out['drawdown'] = 1.0 - close / out['peak']
        out['log_gap'] = np.log(opening / out['previous_close'])
        out['gap_flow'] = out['log_gap'].rolling(params.gap_sum_window, min_periods=params.gap_sum_window).sum()
        prior_flow = out['gap_flow'].shift(1)
        out['gap_upper'] = prior_flow.rolling(params.gap_channel_window, min_periods=params.gap_channel_window).max()
        out['gap_lower'] = prior_flow.rolling(params.gap_channel_window, min_periods=params.gap_channel_window).min()
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        values = indicators[['drawdown', 'gap_flow', 'gap_upper', 'gap_lower', 'atr']].to_numpy(dtype=float)
        opening = data['open'].to_numpy(dtype=float)
        close = data['close'].to_numpy(dtype=float)
        ready = np.isfinite(values).all(axis=1)
        ready &= np.isfinite(opening) & np.isfinite(close) & (opening > 0) & (close > 0)
        ready &= values[:, 4] > 0
        long_entry = ready & (values[:, 0] >= params.long_depth) & (values[:, 1] > np.maximum(values[:, 2], 0.0))
        short_entry = ready & (values[:, 0] <= params.short_depth) & (values[:, 1] < np.minimum(values[:, 3], 0.0))
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry = 0.0
        stop = -np.inf
        best = 0.0
        active = False
        age = 0
        for i in range(len(data)):
            if position:
                if entry == 0.0:
                    # Yesterday's decision fills at today's open.
                    if not ready[i] or i == 0:
                        position = 0
                        continue
                    entry = opening[i]
                    stop = -params.trail_atr * values[i - 1, 4]
                    best = 0.0
                    active = False
                    age = 0
                age += 1
                if not ready[i]:
                    position = 0
                    entry = 0.0
                    continue
                gain = position * (close[i] - entry)
                # Stops are observed at the close; liquidation is next open.
                if gain <= stop or age >= params.hold_bars:
                    position = 0
                    entry = 0.0
                    continue
                best = max(best, gain)
                if best / entry >= params.breakeven_pct:
                    active = True
                if active:
                    # Signed profit-space stop only increases: long price
                    # stops rise, and mirrored short price stops fall.
                    stop = max(stop, 0.0, best - params.trail_atr * values[i, 4])
            elif long_entry[i]:
                position = 1
                entry = 0.0
            elif short_entry[i]:
                position = -1
                entry = 0.0
            signal[i] = position
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = Params(trail_atr=2.0)
    index = pd.date_range('2020-01-01', periods=4)
    for direction in (1, -1):
        close = np.array([100.0, 100.0 + direction * 3.0, 100.0 + direction * 0.5, 100.0 + direction * 0.5])
        data = pd.DataFrame({'open': 100.0, 'high': np.maximum(close, 100.0) + 1.0, 'low': np.minimum(close, 100.0) - 1.0, 'close': close, 'volume': 1000.0}, index=index)
        ind = pd.DataFrame({'drawdown': 0.1 if direction == 1 else 0.0, 'gap_flow': 2.0 * direction, 'gap_upper': 1.0, 'gap_lower': -1.0, 'atr': 1.0}, index=index)
        result = strategy.generate_signals(data, ind, None, params).data
        assert result['signal'].tolist() == [0, direction, direction, 0]
        assert (result['size'] > 0).all()
    ind.loc[:, 'atr'] = np.nan
    assert not strategy.generate_signals(data, ind, None, params).data['signal'].any()


if __name__ == '__main__':
    _self_check()
