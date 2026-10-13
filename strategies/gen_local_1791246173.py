from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    roc_bars: int = 5
    acceleration_bars: int = 3
    efficiency_bars: int = 20
    history_bars: int = 126
    acceleration_percentile: float = 0.85
    efficiency_percentile: float = 0.60
    atr_bars: int = 14
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791246173'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(
            params.roc_bars + params.acceleration_bars + params.history_bars + 1,
            params.efficiency_bars + params.history_bars + 1,
            params.atr_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['previous_close'] = close.shift(1)
        out['roc'] = close.div(close.shift(params.roc_bars)).sub(1)
        out['acceleration'] = out['roc'].diff(params.acceleration_bars)
        out['absolute_acceleration'] = out['acceleration'].abs()
        out['step_distance'] = close.diff().abs()
        out['path_distance'] = out['step_distance'].rolling(params.efficiency_bars).sum()
        out['net_distance'] = close.diff(params.efficiency_bars).abs()
        out['efficiency'] = out['net_distance'].div(out['path_distance'].replace(0, np.nan))
        out['acceleration_threshold'] = out['absolute_acceleration'].shift(1).rolling(params.history_bars).quantile(params.acceleration_percentile)
        out['efficiency_threshold'] = out['efficiency'].shift(1).rolling(params.history_bars).quantile(params.efficiency_percentile)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        required = ['acceleration', 'acceleration_threshold', 'efficiency', 'efficiency_threshold', 'atr']
        out['ready'] = out[required].notna().all(axis=1) & close.notna() & out['atr'].gt(0)
        out['efficient_regime'] = out['efficiency'].gt(out['efficiency_threshold'])
        extreme = out['absolute_acceleration'].gt(out['acceleration_threshold'])
        direction = np.sign(out['acceleration'].fillna(0))
        out['candidate'] = pd.Series(
            np.where(out['efficient_regime'], direction, -direction),
            index=data.index,
        ).where(out['ready'] & extreme, 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        closes = data['close'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        atr = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        pending = False
        entry_bar = -1
        stop = np.nan
        entry_atr = np.nan

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # Entry is observed only on the actual next-bar fill session.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                    entry_bar = i
                else:
                    position = 0
                    continue

            if position:
                invalid = not np.isfinite(closes[i]) or closes[i] <= 0
                stopped = not invalid and position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= params.holding_bars
                if invalid or stopped or expired:
                    position = 0
                    # A stop or expiry cannot immediately re-enter on the same close.
                    continue
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                entry_atr = float(atr[i])
                pending = True

            signals[i] = position

        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=5)
    for direction in (1, -1):
        data = pd.DataFrame({
            'open': [100.0, 100.0 + direction * 10, 100.0 + direction * 11, 100.0 + direction * 12, 100.0],
            'close': [100.0, 100.0 + direction * 11, 100.0 + direction * 9, 100.0 + direction * 7, 100.0],
            'high': 120.0,
            'low': 80.0,
            'volume': 1000.0,
        }, index=index)
        derived = pd.DataFrame({'atr': [1.0, 100.0, 100.0, 100.0, 100.0], 'candidate': [direction, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, StrategyParams(stop_atr=2.0)).data
        assert result['signal'].tolist() == [0, direction, direction, direction, 0]
        assert result['size'].gt(0).all()
        timed = strategy.generate_signals(data, derived, None, StrategyParams(stop_atr=2.0, holding_bars=2)).data
        assert timed['signal'].tolist() == [0, direction, direction, 0, 0]


if __name__ == '__main__':
    _self_check()
