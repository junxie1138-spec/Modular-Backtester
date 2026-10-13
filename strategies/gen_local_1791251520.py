from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 42
    drawdown_memory: int = 10
    pressure_window: int = 7
    min_drawdown: float = 0.03
    recovery_ratio: float = 1.3
    size_floor: float = 0.25


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = 'gen_local_1791251520'

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    @staticmethod
    def warmup_bars(params: RecoveryParams) -> int:
        return int(max(params.peak_window + params.drawdown_memory - 1,
                       2 * params.pressure_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.peak_window, params.drawdown_memory, params.pressure_window) < 2:
            raise ValueError('Window lengths must be at least two.')
        if not (0.0 < params.min_drawdown < 1.0 and
                np.isfinite(params.recovery_ratio) and params.recovery_ratio > 0.0 and
                0.0 < params.size_floor <= 1.0):
            raise ValueError('Invalid drawdown, recovery ratio, or sizing parameter.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        out['change'] = close.diff()
        out['peak'] = close.rolling(params.peak_window).max()
        out['drawdown'] = (1.0 - close / out['peak'].replace(0.0, np.nan)).clip(lower=0.0)
        out['remembered_drawdown'] = out['drawdown'].rolling(params.drawdown_memory).max()
        rising = out['change'].gt(0.0)
        falling = out['change'].lt(0.0)
        out['up_streak'] = rising.groupby((~rising).cumsum()).cumsum().clip(upper=params.pressure_window).astype(float)
        out['down_streak'] = falling.groupby((~falling).cumsum()).cumsum().clip(upper=params.pressure_window).astype(float)
        out['recovery_seed'] = out['up_streak'].rolling(params.pressure_window).max()
        out['recovery_pressure'] = out['up_streak'].rolling(params.pressure_window).sum()
        out['selling_pressure'] = out['down_streak'].rolling(params.pressure_window).sum()
        weighted_selling = params.recovery_ratio * out['selling_pressure']
        out['weighted_selling_pressure'] = weighted_selling
        out['total_pressure'] = out['recovery_pressure'] + weighted_selling
        out['strength'] = ((out['recovery_pressure'] - weighted_selling) /
                           out['total_pressure'].replace(0.0, np.nan)).clip(-1.0, 1.0)
        out['ready'] = np.arange(len(data)) >= GeneratedStrategy.warmup_bars(params) - 1
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        strength = indicators['strength'].fillna(0.0)
        entry_condition = (
            indicators['ready'].astype(bool)
            & indicators['remembered_drawdown'].ge(params.min_drawdown)
            & indicators['recovery_seed'].ge(2.0)
            & strength.gt(0.0)
        )
        df = pd.DataFrame(index=data.index)
        # The position remains long exactly while the entry condition holds.
        # Its reversal is the only exit; there are no independent exit rules.
        df['signal'] = entry_condition.astype(int)
        df['size'] = params.size_floor + (1.0 - params.size_floor) * strength.clip(0.0, 1.0)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(params.size_floor).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    params = RecoveryParams()
    index = pd.date_range('2020-01-01', periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'ready': [True] * 5,
        'remembered_drawdown': [0.05] * 5,
        'recovery_seed': [2.0] * 5,
        'strength': [-0.2, 0.2, 0.8, -0.1, -0.2],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0]
    assert np.allclose(result['size'], [0.25, 0.25, 0.40, 0.85, 0.25])
