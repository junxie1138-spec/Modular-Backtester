from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureChangeParams:
    reaction_window: int = 4
    baseline_window: int = 20
    area_threshold: float = 0.20
    energy_shift_threshold: float = 0.30


class GeneratedStrategy(BaseStrategy[SignatureChangeParams]):
    strategy_id = "gen_local-w2_1791236707"

    @classmethod
    def params_type(cls) -> type[SignatureChangeParams]:
        return SignatureChangeParams

    @staticmethod
    def warmup_bars(params: SignatureChangeParams) -> int:
        return max(2, int(params.reaction_window)) + max(2, int(params.baseline_window)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureChangeParams) -> pd.DataFrame:
        n = max(2, int(params.reaction_window))
        b = max(2, int(params.baseline_window))
        out = pd.DataFrame(index=data.index)
        out['log_close'] = np.log(data['close'].where(data['close'] > 0))
        out['log_return'] = out['log_close'].diff().replace([np.inf, -np.inf], np.nan)
        r = out['log_return']
        out['return_energy'] = r.pow(2)
        out['signed_energy'] = np.sign(r) * out['return_energy']
        out['path_variation'] = r.abs().rolling(n, min_periods=n).sum()
        # Twice the antisymmetric level-two signature of (time, log price).
        out['area_numerator'] = sum((n - 1 - 2 * lag) * r.shift(lag) for lag in range(n))
        scale = (n - 1) * out['path_variation']
        out['path_area'] = out['area_numerator'].div(scale.where(scale > 0)).fillna(0.0)
        out['recent_energy'] = out['return_energy'].rolling(n, min_periods=n).sum()
        out['recent_signed_energy'] = out['signed_energy'].rolling(n, min_periods=n).sum()
        out['baseline_energy'] = out['return_energy'].shift(n).rolling(b, min_periods=b).sum()
        out['baseline_signed_energy'] = out['signed_energy'].shift(n).rolling(b, min_periods=b).sum()
        out['recent_balance'] = out['recent_signed_energy'].div(out['recent_energy'].where(out['recent_energy'] > 0)).fillna(0.0)
        out['baseline_balance'] = out['baseline_signed_energy'].div(out['baseline_energy'].where(out['baseline_energy'] > 0)).fillna(0.0)
        out['energy_shift'] = out['recent_balance'] - out['baseline_balance']
        out['ready'] = out[['area_numerator', 'recent_energy', 'baseline_energy']].notna().all(axis=1)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureChangeParams) -> SignalFrame:
        curvature_agrees = indicators['path_area'].gt(params.area_threshold)
        energy_agrees = indicators['recent_balance'].gt(0.0) & indicators['energy_shift'].gt(params.energy_shift_threshold)
        entry_condition = indicators['ready'].astype(bool) & curvature_agrees & energy_agrees
        df = pd.DataFrame(index=data.index)
        # Remain long while the same AND condition holds; exit when it flips false.
        df['signal'] = entry_condition.fillna(False).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = SignatureChangeParams()
    index = pd.date_range('2020-01-01', periods=64, freq='D')
    returns = np.tile(np.array([-0.004, -0.002, 0.002, 0.006]), 16)
    close = 100.0 * np.exp(np.cumsum(returns))
    data = pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close, 'volume': 1000000.0}, index=index)
    full = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:40], params)
    pd.testing.assert_frame_equal(full.iloc[:40], prefix)
    assert not full.isna().any().any()
    assert not full['ready'].iloc[:params.reaction_window + params.baseline_window].any()
    probe = pd.DataFrame({'path_area': [0.0, 0.5, 0.5, 0.5, 0.0], 'recent_balance': [0.5] * 5, 'energy_shift': [0.5, 0.5, 0.5, 0.0, 0.5], 'ready': [True] * 5}, index=index[:5])
    result = GeneratedStrategy.generate_signals(data.iloc[:5], probe, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    assert result['size'].gt(0).all()
    accelerating = data.iloc[:5].copy()
    accelerating['close'] = 100.0 * np.exp(np.array([0.0, 0.01, 0.03, 0.06, 0.10]))
    area = GeneratedStrategy.indicators(accelerating, params)['path_area'].iloc[-1]
    assert np.isclose(area, 1.0 / 3.0)


if __name__ == '__main__':
    _self_check()
