from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityRecoveryParams:
    graph_bars: int = 12
    peak_bars: int = 63
    percentile_bars: int = 126
    degree_percentile: float = 0.75
    drawdown_percentile: float = 0.65


class GeneratedStrategy(BaseStrategy[VisibilityRecoveryParams]):
    strategy_id = 'gen_local-w2_1791321423'

    @classmethod
    def params_type(cls) -> type[VisibilityRecoveryParams]:
        return VisibilityRecoveryParams

    @staticmethod
    def warmup_bars(params: VisibilityRecoveryParams) -> int:
        return int(max(params.graph_bars, params.peak_bars) + params.percentile_bars)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VisibilityRecoveryParams) -> pd.DataFrame:
        if params.graph_bars < 2 or params.peak_bars < 2 or params.percentile_bars < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not (0.0 < params.degree_percentile < 1.0 and 0.0 < params.drawdown_percentile < 1.0):
            raise ValueError('Percentiles must lie strictly between zero and one.')
        close = data['close'].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        current = close.to_numpy()
        degree = np.zeros(len(data), dtype=float)
        barrier = np.full(len(data), -np.inf)
        # An edge exists when every intervening close is below both endpoints.
        for lag in range(1, params.graph_bars):
            past = close.shift(lag).to_numpy()
            visible = np.isfinite(current) & np.isfinite(past) & (barrier < np.minimum(current, past))
            degree += visible.astype(float)
            barrier = np.maximum(barrier, past)
        complete = close.rolling(params.graph_bars, min_periods=params.graph_bars).count().eq(params.graph_bars)
        out = pd.DataFrame(index=data.index)
        out['degree'] = pd.Series(degree, index=data.index).where(complete)
        out['peak'] = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        out['drawdown'] = 1.0 - close / out['peak']
        out['degree_threshold'] = out['degree'].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.degree_percentile)
        out['drawdown_threshold'] = out['drawdown'].shift(1).rolling(params.percentile_bars, min_periods=params.percentile_bars).quantile(params.drawdown_percentile)
        out['recovering'] = close.gt(close.shift(1))
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityRecoveryParams) -> SignalFrame:
        required = ['degree', 'drawdown', 'degree_threshold', 'drawdown_threshold']
        ready = pd.Series(np.isfinite(indicators[required].to_numpy()).all(axis=1), index=data.index)
        entry = (
            ready
            & indicators['degree'].gt(indicators['degree_threshold'])
            & indicators['drawdown'].gt(indicators['drawdown_threshold'])
            & indicators['recovering'].fillna(False).astype(bool)
        )
        df = pd.DataFrame(index=data.index)
        # Remain long while the entry predicate holds; its reversal exits.
        # Daily topology bursts target 1-2 sessions without imposing a time exit.
        df['signal'] = entry.fillna(False).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = VisibilityRecoveryParams(graph_bars=6, peak_bars=6, percentile_bars=4)
    index = pd.date_range('2020-01-01', periods=6)
    prices = [5.0, 4.0, 3.0, 2.0, 1.0, 4.5]
    data = pd.DataFrame({name: prices for name in ['open', 'high', 'low', 'close']}, index=index)
    data['volume'] = 100.0
    topology = GeneratedStrategy.indicators(data, params)
    assert topology['degree'].iloc[-1] == 5.0
    assert topology['degree'].iloc[:-1].isna().all()
    probe = pd.DataFrame({'degree': 3.0, 'degree_threshold': 2.0, 'drawdown': 0.1, 'drawdown_threshold': 0.05, 'recovering': [True, True, False, True, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(data, probe, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, 1, 0]
    assert result['size'].gt(0.0).all()
    probe.loc[index[0], 'degree_threshold'] = np.nan
    assert GeneratedStrategy.generate_signals(data, probe, None, params).data['signal'].iloc[1] == 0
    assert GeneratedStrategy.warmup_bars(params) == 10


if __name__ == '__main__':
    _self_check()
