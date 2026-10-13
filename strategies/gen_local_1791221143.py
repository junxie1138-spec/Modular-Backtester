from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 20
    stress_window: int = 63
    rank_window: int = 5
    recovery_window: int = 4
    stress_threshold: float = 0.65
    lead_threshold: float = 0.10


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791221143'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(
            params.peak_window + params.stress_window - 1,
            params.rank_window + params.recovery_window - 1,
        ) + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.peak_window, params.stress_window, params.rank_window, params.recovery_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < params.stress_threshold < 1.0 or not 0.0 < params.lead_threshold < 1.0:
            raise ValueError('Rank thresholds must lie strictly between zero and one.')
        close = data['close'].where(data['close'] > 0)
        low = data['low'].where(data['low'] > 0)
        out = pd.DataFrame(index=data.index)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = (1.0 - close / out['peak']).clip(lower=0.0)
        out['stress_rank'] = out['drawdown'].rolling(
            params.stress_window, min_periods=params.stress_window,
        ).rank(method='average', pct=True)
        out['low_rank'] = low.rolling(
            params.rank_window, min_periods=params.rank_window,
        ).rank(method='average', pct=True)
        out['close_rank'] = close.rolling(
            params.rank_window, min_periods=params.rank_window,
        ).rank(method='average', pct=True)
        out['rank_gap'] = out['low_rank'] - out['close_rank']
        out['rank_lead'] = out['rank_gap'].rolling(
            params.recovery_window, min_periods=params.recovery_window,
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoveryParams,
    ) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        entry_condition = (
            ready
            & indicators['drawdown'].gt(0.0)
            & indicators['stress_rank'].ge(params.stress_threshold)
            & indicators['rank_lead'].gt(params.lead_threshold)
        )
        df = pd.DataFrame(index=data.index)
        # Maintain exposure exactly while the entry condition remains true.
        # Its reversal is the sole exit; no time expiry or price stop applies.
        df['signal'] = entry_condition.astype(int)
        conviction = (
            indicators['stress_rank']
            * (indicators['rank_lead'] / 0.40).clip(lower=0.0, upper=1.0)
        )
        df['size'] = (0.25 + 0.70 * conviction).fillna(0.25).clip(0.25, 0.95)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    rng = np.random.default_rng(17)
    index = pd.date_range('2020-01-01', periods=500, freq='B')
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(index))))
    data = pd.DataFrame({
        'open': close,
        'high': close * 1.01,
        'low': close * (1.0 - rng.uniform(0.001, 0.025, len(index))),
        'close': close,
        'volume': 1000000.0,
    }, index=index)
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    ind = strategy.indicators(data, params)
    result = strategy.generate_signals(data, ind, None, params).data
    raw = (
        ind.notna().all(axis=1)
        & ind['drawdown'].gt(0.0)
        & ind['stress_rank'].ge(params.stress_threshold)
        & ind['rank_lead'].gt(params.lead_threshold)
    )
    assert result['signal'].equals(raw.astype(int).shift(1).fillna(0).astype(int))
    assert result['signal'].isin([0, 1]).all()
    assert result['size'].between(0.25, 0.95).all()
    assert result['signal'].sum() > 0
    assert result['signal'].iloc[:strategy.warmup_bars(params) - 1].eq(0).all()
    prefix = data.iloc[:300]
    prior = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(prior, result.iloc[:300])


if __name__ == '__main__':
    _self_check()
