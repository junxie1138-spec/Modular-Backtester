from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    profit_target: float = 0.04
    holding_bars: int = 18

    def __post_init__(self):
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError('profit_target must be finite and positive')
        if not isinstance(self.holding_bars, int) or not 15 <= self.holding_bars <= 20:
            raise ValueError('holding_bars must be an integer from 15 to 20')


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1791253187'

    @classmethod
    def params_type(cls):
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        result = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        opening = data['open'].astype(float).replace([np.inf, -np.inf], np.nan)
        result['close'] = close.where(close > 0)
        result['open'] = opening.where(opening > 0)
        result['previous_close'] = result['close'].shift(1)
        result['gap'] = result['open'].div(result['previous_close']).sub(1)
        result['repaired'] = result['gap'].lt(0) & result['close'].ge(result['previous_close'])
        repaired = result['repaired']
        previous = repaired.shift(1, fill_value=False)
        result['repair_streak'] = (
            repaired.astype(int)
            + (repaired & previous).astype(int)
            + (repaired & previous & repaired.shift(2, fill_value=False)).astype(int)
        )
        result['entry'] = (
            result['repair_streak'].shift(1).eq(3)
            & result['gap'].gt(0)
            & result['close'].ge(result['open'])
        ).fillna(False)
        return result

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        opening = indicators['open'].to_numpy(dtype=float)
        close = indicators['close'].to_numpy(dtype=float)
        entry = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            # A decision on the preceding close fills at this session's open.
            if active and i == entry_bar:
                entry_price = opening[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False

            if active:
                target_hit = (
                    np.isfinite(close[i])
                    and close[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_expired = i - entry_bar + 1 >= params.holding_bars
                if target_hit or time_expired:
                    active = False
            elif entry[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan

            signals[i] = int(active)

        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = StrategyParams()
    index = pd.date_range('2020-01-01', periods=32, freq='B')
    data = pd.DataFrame({
        'open': 100.0, 'high': 106.0, 'low': 98.0,
        'close': 100.0, 'volume': 1000000.0,
    }, index=index)
    data.iloc[1:4, data.columns.get_loc('open')] = 99.0
    data.iloc[4, data.columns.get_loc('open')] = 100.5
    data.iloc[4, data.columns.get_loc('close')] = 101.0
    def run():
        derived = GeneratedStrategy.indicators(data, params)
        return GeneratedStrategy.generate_signals(data, derived, None, params).data['signal']
    signal = run()
    assert signal.iloc[:5].eq(0).all()
    assert signal.iloc[5:23].eq(1).all() and signal.iloc[23:].eq(0).all()
    data.iloc[6, data.columns.get_loc('close')] = 105.0
    signal = run()
    assert signal.iloc[5:7].eq(1).all() and signal.iloc[7:].eq(0).all()


if __name__ == '__main__':
    _self_check()
