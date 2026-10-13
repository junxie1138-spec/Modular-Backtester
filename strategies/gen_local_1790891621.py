from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DiffusionParams:
    roc_bars: int = 5
    smoothing_bars: int = 15
    participation_bars: int = 20
    vol_bars: int = 20
    target_vol: float = 0.12
    entry_score: float = 0.04
    strength_scale: float = 0.20


class GeneratedStrategy(BaseStrategy[DiffusionParams]):
    strategy_id = 'gen_local_1790891621'

    @classmethod
    def params_type(cls):
        return DiffusionParams

    def warmup_bars(self, params: DiffusionParams) -> int:
        return int(max(
            2 * params.roc_bars + params.smoothing_bars,
            params.participation_bars + 1,
            params.vol_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: DiffusionParams) -> pd.DataFrame:
        if min(params.roc_bars, params.smoothing_bars, params.participation_bars, params.vol_bars) < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not (
            np.isfinite(params.target_vol) and params.target_vol > 0
            and np.isfinite(params.strength_scale) and params.strength_scale > 0
            and np.isfinite(params.entry_score) and params.entry_score >= 0
        ):
            raise ValueError('Invalid volatility target or signal thresholds.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['roc'] = close.pct_change(params.roc_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['acceleration'] = out['roc'].diff(params.roc_bars)
        out['smooth_acceleration'] = out['acceleration'].rolling(params.smoothing_bars).mean()
        out['daily_vol'] = out['return'].rolling(params.vol_bars).std(ddof=0)
        up = out['return'].gt(0).astype(float).where(out['return'].notna())
        out['susceptible'] = 1.0 - up.rolling(params.participation_bars).mean()
        out['annual_vol'] = out['daily_vol'] * np.sqrt(252.0)
        out['score'] = (
            out['smooth_acceleration'] * out['susceptible']
            / (out['daily_vol'].replace(0.0, np.nan) * np.sqrt(params.roc_bars))
        )
        strength = out['score'].clip(lower=0.0)
        out['conviction'] = strength / (strength + params.strength_scale)
        out['target_size'] = (
            params.target_vol / out['annual_vol'].replace(0.0, np.nan)
            * out['conviction']
        ).clip(0.01, 0.95).fillna(0.01)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DiffusionParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Fifteen-session smoothing targets multiweek diffusion episodes.
        # Entry and reversal-only exit use exactly the same condition.
        df['signal'] = indicators['score'].gt(params.entry_score).astype(int)
        df['size'] = indicators['target_size'].shift(1).fillna(0.01).astype(float)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = DiffusionParams()
    strategy = GeneratedStrategy()
    t = np.arange(320, dtype=float)
    close = 100.0 * np.exp(np.cumsum(0.001 + 0.03 * np.sin(t / 14.0)))
    data = pd.DataFrame(
        {'open': close, 'high': close * 1.01, 'low': close * 0.99,
         'close': close, 'volume': 1000000.0},
        index=pd.date_range('2020-01-01', periods=len(t), freq='B'),
    )
    derived = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, derived, None, params).data
    expected = derived['score'].gt(params.entry_score).astype(int).shift(1).fillna(0).astype(int)
    assert signals['signal'].equals(expected)
    assert signals.index.equals(data.index) and np.isfinite(derived.to_numpy()).all()
    assert set(signals['signal']) == {0, 1}
    assert signals['size'].between(0.01, 0.95).all()
    assert signals['size'].equals(derived['target_size'].shift(1).fillna(0.01))
    assert signals.loc[signals['signal'].eq(1), 'size'].nunique() > 1
    assert signals['signal'].iloc[:strategy.warmup_bars(params)].eq(0).all()
    data.loc[:, ['open', 'high', 'low', 'close']] = 100.0
    flat = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert flat['signal'].eq(0).all() and flat['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
