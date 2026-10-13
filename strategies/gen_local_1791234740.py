from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DepthParticipationParams:
    anchor_window: int = 20
    pressure_window: int = 6
    trend_window: int = 12
    atr_window: int = 14
    depth_ratio: float = 0.8
    max_depth_atr: float = 1.0
    breakeven_pct: float = 0.008
    trail_atr: float = 1.4
    initial_stop_atr: float = 2.0
    holding_bars: int = 4

    def __post_init__(self):
        if min(self.anchor_window, self.pressure_window, self.trend_window, self.atr_window) < 2:
            raise ValueError('Indicator windows must be at least two bars')
        if not 0.0 < self.depth_ratio < 1.0:
            raise ValueError('depth_ratio must be between zero and one')
        if not all(np.isfinite(x) and x > 0.0 for x in (self.max_depth_atr, self.breakeven_pct, self.trail_atr, self.initial_stop_atr)):
            raise ValueError('Depth and stop parameters must be finite and positive')
        if not 3 <= self.holding_bars <= 5:
            raise ValueError('holding_bars must be between three and five')


class GeneratedStrategy(BaseStrategy[DepthParticipationParams]):
    strategy_id = 'gen_local_1791234740'

    @classmethod
    def params_type(cls):
        return DepthParticipationParams

    @staticmethod
    def warmup_bars(params: DepthParticipationParams) -> int:
        return int(max(params.anchor_window + params.pressure_window, params.atr_window + 2, params.trend_window + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: DepthParticipationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out.loc[out['previous_close'].isna(), 'true_range'] = np.nan
        out['atr'] = out['true_range'].rolling(params.atr_window).mean()
        out['peak'] = close.rolling(params.anchor_window).max()
        out['trough'] = close.rolling(params.anchor_window).min()
        denominator = out['atr'].where(out['atr'] > 0.0)
        out['drawdown'] = (out['peak'] - close) / denominator
        out['drawup'] = (close - out['trough']) / denominator
        out['usable_volume'] = data['volume'].where(data['volume'] >= 0.0)
        out['volume_sum'] = out['usable_volume'].rolling(params.pressure_window).sum()
        volume_sum = out['volume_sum'].where(out['volume_sum'] > 0.0)
        for depth in ('drawdown', 'drawup'):
            out[depth + '_volume'] = out[depth] * out['usable_volume']
            out[depth + '_mean'] = out[depth].rolling(params.pressure_window).mean()
            out[depth + '_weighted'] = out[depth + '_volume'].rolling(params.pressure_window).sum() / volume_sum
        out['displacement'] = close.diff(params.trend_window)
        needed = ['atr', 'drawdown_mean', 'drawup_mean', 'drawdown_weighted', 'drawup_weighted', 'displacement']
        out['valid'] = np.isfinite(out[needed]).all(axis=1) & (out['atr'] > 0.0)
        out['long_candidate'] = (
            out['valid']
            & (out['displacement'] > 0.0)
            & (out['drawdown_mean'] > 0.0)
            & (out['drawdown_weighted'] < params.depth_ratio * out['drawdown_mean'])
            & (out['drawdown_weighted'] <= params.max_depth_atr)
        )
        out['short_candidate'] = (
            out['valid']
            & (out['displacement'] < 0.0)
            & (out['drawup_mean'] > 0.0)
            & (out['drawup_weighted'] < params.depth_ratio * out['drawup_mean'])
            & (out['drawup_weighted'] <= params.max_depth_atr)
        )
        out['long_confirmed'] = out['long_candidate'] & out['long_candidate'].shift(1, fill_value=False)
        out['short_confirmed'] = out['short_candidate'] & out['short_candidate'].shift(1, fill_value=False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DepthParticipationParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        valid = indicators['valid'].to_numpy(dtype=bool)
        longs = indicators['long_confirmed'].to_numpy(dtype=bool)
        shorts = indicators['short_confirmed'].to_numpy(dtype=bool)
        side = 0
        pending = False
        entry = 0.0
        entry_atr = 0.0
        best = 0.0
        stop = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            if side:
                if pending:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        side = 0
                        pending = False
                        continue
                    # The preceding close's decision fills at this bar's open.
                    entry = opens[i]
                    best = side * entry
                    stop = best - params.initial_stop_atr * entry_atr
                    pending = False
                age += 1
                if not valid[i] or not np.isfinite(closes[i]) or closes[i] <= 0.0:
                    side = 0
                else:
                    # Signed prices make the protective stop increase for both directions.
                    signed_close = side * closes[i]
                    best = max(best, signed_close)
                    if best - side * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry, best - params.trail_atr * atr[i])
                    # Close-observed exits execute at the next open after the mandatory shift.
                    if signed_close <= stop or age >= params.holding_bars:
                        side = 0
            elif valid[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                if longs[i]:
                    side = 1
                elif shorts[i]:
                    side = -1
                if side:
                    pending = True
                    entry_atr = atr[i]
                    armed = False
                    age = 0
            decisions[i] = side
        df = pd.DataFrame({'signal': decisions, 'size': np.full(len(data), 0.95, dtype=float)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = DepthParticipationParams(holding_bars=5)
    index = pd.date_range('2020-01-01', periods=6)
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0])
        data = pd.DataFrame({'open': 100.0, 'high': close + 1.0, 'low': close - 1.0, 'close': close, 'volume': 1000.0}, index=index)
        entry_flags = np.array([False, True, False, False, False, False])
        indicators = pd.DataFrame({'atr': 1.0, 'valid': True, 'long_confirmed': entry_flags if direction == 1 else False, 'short_confirmed': entry_flags if direction == -1 else False}, index=index)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result['signal'].tolist() == [0, 0, direction, direction, direction, 0]
        assert (result['size'] > 0.0).all()
        derived = GeneratedStrategy.indicators(data, params)
        assert not derived['long_confirmed'].any()
        assert not derived['short_confirmed'].any()
        for name in ('long', 'short'):
            expected = derived[name + '_candidate'] & derived[name + '_candidate'].shift(1, fill_value=False)
            assert derived[name + '_confirmed'].equals(expected)


if __name__ == '__main__':
    _self_check()
