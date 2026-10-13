from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockRecoveryParams:
    shock_bars: int = 10
    repair_bars: int = 3
    peak_bars: int = 30
    min_drawdown: float = 0.02
    max_drawdown: float = 0.15
    min_shock_loss: float = 0.025
    concentration_threshold: float = 0.55
    seller_volume_ratio: float = 0.8
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ShockRecoveryParams]):
    strategy_id = 'gen_local_1791238757'

    @classmethod
    def params_type(cls) -> type[ShockRecoveryParams]:
        return ShockRecoveryParams

    def warmup_bars(self, params: ShockRecoveryParams) -> int:
        return int(max(params.peak_bars, params.shock_bars + params.repair_bars + 1))

    def indicators(self, data: pd.DataFrame, params: ShockRecoveryParams) -> pd.DataFrame:
        if min(params.shock_bars, params.repair_bars, params.peak_bars, params.hold_bars) < 1:
            raise ValueError('Lookbacks and holding period must be positive.')
        if not (0 < params.min_drawdown < params.max_drawdown < 1):
            raise ValueError('Drawdown bounds must satisfy 0 < minimum < maximum < 1.')
        if not (0 < params.concentration_threshold <= 1):
            raise ValueError('Concentration threshold must lie in (0, 1].')
        if not (0 < params.min_shock_loss < 1 and params.seller_volume_ratio > 0):
            raise ValueError('Shock loss and volume ratio must be positive and valid.')

        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['log_return'] = np.log1p(out['return'])
        out['downside_energy'] = out['return'].clip(upper=0).pow(2)
        out['down_day'] = out['return'].lt(0).astype(float).where(out['return'].notna())
        volume = data['volume'].where(data['volume'] >= 0)
        out['down_volume'] = volume * out['down_day']

        shock = params.shock_bars
        repair = params.repair_bars
        out['shock_log_return'] = out['log_return'].rolling(shock, min_periods=shock).sum().shift(repair)
        out['shock_energy'] = out['downside_energy'].rolling(shock, min_periods=shock).sum().shift(repair)
        out['largest_shock_energy'] = out['downside_energy'].rolling(shock, min_periods=shock).max().shift(repair)
        out['concentration'] = out['largest_shock_energy'] / out['shock_energy'].where(out['shock_energy'] > 0)
        out['repair_log_return'] = out['log_return'].rolling(repair, min_periods=repair).sum()
        out['rolling_peak'] = close.rolling(params.peak_bars, min_periods=params.peak_bars).max()
        out['drawdown'] = close / out['rolling_peak'] - 1.0

        out['shock_down_count'] = out['down_day'].rolling(shock, min_periods=shock).sum().shift(repair)
        out['repair_down_count'] = out['down_day'].rolling(repair, min_periods=repair).sum()
        out['shock_down_volume'] = out['down_volume'].rolling(shock, min_periods=shock).sum().shift(repair)
        out['repair_down_volume'] = out['down_volume'].rolling(repair, min_periods=repair).sum()
        out['shock_mean_down_volume'] = out['shock_down_volume'] / out['shock_down_count'].where(out['shock_down_count'] > 0)
        out['repair_mean_down_volume'] = out['repair_down_volume'] / out['repair_down_count'].where(out['repair_down_count'] > 0)
        out['volume_ratio'] = out['repair_mean_down_volume'] / out['shock_mean_down_volume'].where(out['shock_mean_down_volume'] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockRecoveryParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError('Holding period must be positive.')
        ind = indicators
        return_agreement = (
            ind['drawdown'].between(-params.max_drawdown, -params.min_drawdown)
            & ind['shock_log_return'].le(np.log1p(-params.min_shock_loss))
            & ind['concentration'].ge(params.concentration_threshold)
            & ind['repair_log_return'].gt(0)
        )
        volume_agreement = (
            ind['repair_down_count'].gt(0)
            & ind['volume_ratio'].le(params.seller_volume_ratio)
        )
        required = ['drawdown', 'shock_log_return', 'concentration', 'repair_log_return', 'repair_down_count', 'volume_ratio']
        ready = np.isfinite(ind[required]).all(axis=1)
        entries = (return_agreement & volume_agreement & ready).fillna(False).to_numpy(dtype=bool)

        raw = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, enter in enumerate(entries):
            if remaining > 0:
                remaining -= 1
            elif enter:
                remaining = params.hold_bars
            raw[i] = int(remaining > 0)

        df = pd.DataFrame({'signal': raw, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = ShockRecoveryParams()
    n = params.hold_bars
    index = pd.date_range('2020-01-01', periods=2 * n + 3, freq='B')
    data = pd.DataFrame(index=index)
    ind = pd.DataFrame({
        'drawdown': -0.05,
        'shock_log_return': -0.04,
        'concentration': 0.9,
        'repair_log_return': 0.01,
        'repair_down_count': 1.0,
        'volume_ratio': 0.5,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result['signal'].tolist() == [0] + [1] * n + [0] + [1] * n + [0]
    ind.loc[index[1:], 'concentration'] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result['signal'].tolist() == [0] + [1] * n + [0] * (n + 2)
    assert result['size'].gt(0).all()
