from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 40
    recent_window: int = 5
    prior_window: int = 15
    recovery_window: int = 3
    atr_window: int = 14
    min_drawdown: float = 0.02
    max_drawdown: float = 0.15
    energy_ratio: float = 0.60
    min_recovery: float = 0.004
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791332355"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.peak_window,
                       params.recent_window + params.prior_window + 1,
                       params.recovery_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['downside_square'] = out['return'].clip(upper=0).pow(2)
        out['peak'] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out['drawdown'] = (1.0 - close / out['peak'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['recent_energy'] = out['downside_square'].rolling(params.recent_window, min_periods=params.recent_window).mean()
        out['prior_energy'] = out['downside_square'].shift(params.recent_window).rolling(params.prior_window, min_periods=params.prior_window).mean()
        out['recovery'] = out['return'].rolling(params.recovery_window, min_periods=params.recovery_window).sum()
        out['previous_close'] = close.shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1, skipna=False)
        out['atr'] = out['true_range'].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out['setup'] = (
            out['drawdown'].between(params.min_drawdown, params.max_drawdown)
            & out['prior_energy'].gt(0)
            & out['recent_energy'].le(params.energy_ratio * out['prior_energy'])
            & out['recovery'].ge(params.min_recovery)
            & out['return'].gt(0)
            & out['atr'].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        setup = indicators['setup'].fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1, fill_value=False)).to_numpy()
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        held = 0

        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                held += 1
                stopped = False
                if valid:
                    # First in-trade close is on the bar after the entry decision.
                    high_water = close if np.isnan(high_water) else max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = close <= stop
                if stopped or held >= params.max_hold:
                    active = False
                    high_water = np.nan
                    stop = -np.inf
                    held = 0
            elif valid and confirmed[i]:
                active = True
                high_water = np.nan
                stop = -np.inf
                held = 0
            decisions[i] = int(active)

        df = pd.DataFrame({'signal': decisions, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({'close': [100., 101., 103., 102., 99., 100.]}, index=index)
    derived = pd.DataFrame({'setup': [True, True, False, False, False, False],
                            'atr': [1.] * 6}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, RecoveryParams(trail_atr=2.0))
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 1, 0]
    # ATR expansion cannot lower an established stop.
    expanded = derived.copy()
    expanded.loc[index[4], 'atr'] = 10.0
    result = strategy.generate_signals(data, expanded, None, RecoveryParams(trail_atr=2.0))
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 1, 0]
    result = strategy.generate_signals(data, derived, None, RecoveryParams(trail_atr=2.0, max_hold=2))
    assert result.data['signal'].tolist() == [0, 0, 1, 1, 0, 0]
