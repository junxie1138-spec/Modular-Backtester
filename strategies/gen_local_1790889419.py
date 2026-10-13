from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRegimeParams:
    complexity_window: int = 18
    complexity_cut: float = 0.45
    pressure_cut: float = 0.20


class GeneratedStrategy(BaseStrategy[OrdinalRegimeParams]):
    strategy_id = "gen_local_1790889419"

    @classmethod
    def params_type(cls) -> type[OrdinalRegimeParams]:
        return OrdinalRegimeParams

    def warmup_bars(self, params: OrdinalRegimeParams) -> int:
        return int(params.complexity_window) + 4

    def indicators(self, data: pd.DataFrame, params: OrdinalRegimeParams) -> pd.DataFrame:
        window = int(params.complexity_window)
        if window < 6:
            raise ValueError("complexity_window must be at least 6")
        if not 0.0 < params.complexity_cut < 1.0:
            raise ValueError("complexity_cut must be between zero and one")
        if not 0.0 <= params.pressure_cut < 1.0:
            raise ValueError("pressure_cut must be in [0, 1)")

        out = pd.DataFrame(index=data.index)
        returns = data['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        triples = np.column_stack((returns.shift(2), returns.shift(1), returns))
        valid_pattern = np.isfinite(triples).all(axis=1)
        order = np.argsort(triples, axis=1, kind='stable')
        code = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        pattern = pd.Series(code.astype(float), index=data.index).where(valid_pattern)
        previous = pattern.shift(1)
        valid_pair = pattern.notna() & previous.notna()
        states = (5, 7, 11, 15, 19, 21)
        joint_entropy = pd.Series(0.0, index=data.index)
        previous_entropy = pd.Series(0.0, index=data.index)

        for state in states:
            frequency = (previous.eq(state).astype(float).where(valid_pair)
                         .rolling(window, min_periods=window).mean())
            previous_entropy -= frequency * np.log(frequency.where(frequency > 0.0, 1.0))
            for next_state in states:
                frequency = ((previous.eq(state) & pattern.eq(next_state))
                             .astype(float).where(valid_pair)
                             .rolling(window, min_periods=window).mean())
                joint_entropy -= frequency * np.log(frequency.where(frequency > 0.0, 1.0))

        out['return'] = returns
        out['ordinal_state'] = pattern
        out['previous_ordinal_state'] = previous
        out['joint_entropy'] = joint_entropy
        out['previous_entropy'] = previous_entropy
        out['complexity'] = ((joint_entropy - previous_entropy) / np.log(6.0)).clip(0.0, 1.0)
        spread = (data['high'] - data['low']).where(lambda values: values > 0.0)
        out['closing_pressure'] = ((2.0 * data['close'] - data['high'] - data['low']) / spread).clip(-1.0, 1.0)
        out['regime_direction'] = pd.Series(
            np.where(out['complexity'] <= params.complexity_cut, 1.0, -1.0),
            index=data.index,
        ).where(out['complexity'].notna())
        out['score'] = out['regime_direction'] * out['closing_pressure']
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalRegimeParams,
    ) -> SignalFrame:
        # One predicate governs entry, continued exposure, and reversal exit.
        # Daily reevaluation targets brief holds without a separate expiry.
        qualified = indicators['score'].gt(params.pressure_cut)
        df = pd.DataFrame(index=data.index)
        df['signal'] = qualified.fillna(False).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
