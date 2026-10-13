from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ConcentrationParams:
    lookback: int = 8
    threshold: float = 0.75

    def __post_init__(self):
        if self.lookback < 2:
            raise ValueError('lookback must be at least 2')
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError('threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[ConcentrationParams]):
    strategy_id = 'gen_local_1791253314'

    @classmethod
    def params_type(cls) -> type[ConcentrationParams]:
        return ConcentrationParams

    @staticmethod
    def warmup_bars(params: ConcentrationParams) -> int:
        return int(params.lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: ConcentrationParams) -> pd.DataFrame:
        n = params.lookback
        out = pd.DataFrame(index=data.index)
        out['clean_close'] = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['clean_close'] = out['clean_close'].where(out['clean_close'] > 0)
        out['clean_volume'] = data['volume'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['clean_volume'] = out['clean_volume'].where(out['clean_volume'] >= 0)
        out['ma'] = out['clean_close'].rolling(n, min_periods=n).mean().shift(1)
        out['price_std'] = out['clean_close'].rolling(n, min_periods=n).std(ddof=0).shift(1)
        out['volume_squared'] = out['clean_volume'].pow(2)
        out['volume_sum'] = out['clean_volume'].rolling(n, min_periods=n).sum().shift(1)
        out['volume_square_sum'] = out['volume_squared'].rolling(n, min_periods=n).sum().shift(1)
        out['effective_sessions'] = (
            out['volume_sum'].pow(2)
            / out['volume_square_sum'].where(out['volume_square_sum'] > 0)
        ).clip(lower=1.0, upper=float(n))
        out['reference_reliability'] = np.sqrt(out['effective_sessions'] / n)
        out['z_score'] = (
            (out['clean_close'] - out['ma'])
            / out['price_std'].where(out['price_std'] > 0)
        )
        out['adjusted_z'] = out['z_score'] * out['reference_reliability']
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ConcentrationParams,
    ) -> SignalFrame:
        z = indicators['adjusted_z'].reindex(data.index)
        entry = np.select(
            [z.le(-params.threshold), z.ge(params.threshold)],
            [1, -1],
            default=0,
        ).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(entry):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # An opposite entry condition exits; reopening requires another bar.
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = positions
        df['size'] = 1.0
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = ConcentrationParams()
    index = pd.date_range('2020-01-01', periods=9)
    data = pd.DataFrame(
        {'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 0.0},
        index=index,
    )
    controlled = pd.DataFrame(
        {'adjusted_z': [np.nan, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, -1.0]},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, controlled, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    flat = GeneratedStrategy.indicators(data, params)
    assert flat.index.equals(data.index)
    assert GeneratedStrategy.generate_signals(data, flat, None, params).data['signal'].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) == params.lookback + 1


if __name__ == '__main__':
    _self_check()
