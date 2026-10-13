from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    window: int = 8
    min_drawdown: float = 0.01
    min_triangles: int = 1
    recovery_floor: float = 0.15
    recovery_ceiling: float = 0.80


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791334984'

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        if not 4 <= params.window <= 10:
            raise ValueError('window must be between 4 and 10')
        return int(params.window)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        w = self.warmup_bars(params)
        if not 0.0 < params.min_drawdown < 1.0:
            raise ValueError('min_drawdown must be between zero and one')
        if params.min_triangles < 1:
            raise ValueError('min_triangles must be positive')
        if not 0.0 <= params.recovery_floor < params.recovery_ceiling <= 1.0:
            raise ValueError('invalid recovery interval')
        log_close = np.log(data['close'].where(data['close'] > 0))
        out = pd.DataFrame(index=data.index)
        for lag in range(w):
            out[f'log_lag_{lag}'] = log_close.shift(lag)
        x = out.to_numpy(dtype=float)
        valid = np.isfinite(x).all(axis=1)
        x = np.where(np.isfinite(x), x, 0.0)
        age = np.argmin(x, axis=1)
        trough = x.min(axis=1)
        older = np.arange(w)[None, :] > age[:, None]
        peak = np.max(np.where(older, x, -np.inf), axis=1)
        peak = np.where(np.isfinite(peak), peak, np.nan)
        depth = 1.0 - np.exp(trough - peak)
        recovery = np.divide(
            x[:, 0] - trough,
            peak - trough,
            out=np.full(len(data), np.nan),
            where=peak > trough,
        )
        edges = {}
        for i in range(w):
            for j in range(i + 1, w):
                edge = valid.copy()
                for k in range(i + 1, j):
                    chord = x[:, i] + (x[:, j] - x[:, i]) * (k - i) / (j - i)
                    edge &= x[:, k] < chord
                edges[i, j] = edge
                out[f'edge_{i}_{j}'] = np.where(valid, edge.astype(float), np.nan)
        triangles = np.zeros(len(data), dtype=float)
        for i in range(1, w):
            for j in range(i + 1, w):
                triangles += (
                    edges[0, i] & edges[0, j] & edges[i, j]
                    & (i > age) & (j > age)
                )
        out['trough_age'] = np.where(valid, age, np.nan)
        out['trough_log'] = np.where(valid, trough, np.nan)
        out['shoulder_log'] = np.where(valid, peak, np.nan)
        out['drawdown_depth'] = np.where(valid, depth, np.nan)
        out['recovery_fraction'] = np.where(valid, recovery, np.nan)
        out['cross_trough_triangles'] = np.where(valid, triangles, np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        entry = (
            indicators['trough_age'].gt(0)
            & indicators['drawdown_depth'].ge(params.min_drawdown)
            & indicators['recovery_fraction'].ge(params.recovery_floor)
            & indicators['recovery_fraction'].le(params.recovery_ceiling)
            & indicators['cross_trough_triangles'].ge(params.min_triangles)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Predicate reversal is the sole exit; no fixed holding-time exit.
        df['signal'] = entry.astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 1.0
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    close = np.array([100.5, 100, 99, 96, 94, 95, 97, 98, 99, 100], dtype=float)
    data = pd.DataFrame(
        {'open': close, 'high': close + 0.5, 'low': close - 0.5,
         'close': close, 'volume': np.full(len(close), 1000000.0)},
        index=pd.date_range('2020-01-01', periods=len(close), freq='B'),
    )
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=8)
    indicators = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, indicators, ctx, params).data
    assert signals['signal'].iloc[:8].eq(0).all()
    assert signals['signal'].iloc[8] == 1
    assert signals['signal'].iloc[9] == 0
    assert signals['signal'].isin([0, 1]).all()
    assert signals['size'].gt(0).all()
    altered = data.copy()
    altered.iloc[-1, altered.columns.get_loc('close')] = 50.0
    changed = strategy.indicators(altered, params)
    pd.testing.assert_frame_equal(indicators.iloc[:-1], changed.iloc[:-1])
