from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    value_window: int = 63
    recovery_window: int = 18
    rank_window: int = 126
    entry_percentile: float = 0.35
    accumulation_floor: float = 0.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local-w2_1791324707'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return int(max(params.value_window, params.recovery_window) + params.rank_window)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.value_window, params.recovery_window, params.rank_window) < 2:
            raise ValueError('Lookback windows must be at least two bars.')
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError('entry_percentile must be between zero and one.')
        if not -1.0 <= params.accumulation_floor <= 1.0:
            raise ValueError('accumulation_floor must be between minus one and one.')
        out = pd.DataFrame(index=data.index)
        volume = data['volume'].clip(lower=0.0)
        out['typical_price'] = (data['high'] + data['low'] + data['close']) / 3.0
        out['value_volume'] = volume.rolling(params.value_window).sum()
        out['value_notional'] = (out['typical_price'] * volume).rolling(params.value_window).sum()
        out['volume_value'] = out['value_notional'] / out['value_volume'].replace(0.0, np.nan)
        out['mean_close'] = data['close'].rolling(params.recovery_window).mean()
        out['discount'] = out['mean_close'] / out['volume_value'].replace(0.0, np.nan) - 1.0
        out['discount_percentile'] = out['discount'].rolling(params.rank_window).rank(pct=True)
        out['daily_range'] = data['high'] - data['low']
        out['close_location'] = ((2.0 * data['close'] - data['high'] - data['low']) / out['daily_range'].where(out['daily_range'] > 0.0)).clip(-1.0, 1.0)
        out.loc[out['daily_range'].eq(0.0), 'close_location'] = 0.0
        out['signed_volume'] = out['close_location'] * volume
        out['recovery_volume'] = volume.rolling(params.recovery_window).sum()
        out['accumulation'] = out['signed_volume'].rolling(params.recovery_window).sum() / out['recovery_volume'].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        # The position exits only when this same entry conjunction becomes false.
        entry = (
            indicators['discount_percentile'].le(params.entry_percentile)
            & indicators['discount'].lt(0.0)
            & indicators['accumulation'].gt(params.accumulation_floor)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        df['signal'] = entry.astype(int)
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    data = pd.DataFrame(index=pd.date_range('2020-01-01', periods=6))
    features = pd.DataFrame({
        'discount_percentile': [np.nan, 0.2, 0.2, 0.6, 0.1, 0.1],
        'discount': [np.nan, -0.02, -0.02, -0.01, -0.03, -0.03],
        'accumulation': [np.nan, 0.1, -0.1, 0.2, 0.2, 0.2],
    }, index=data.index)
    result = GeneratedStrategy.generate_signals(data, features, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 0, 0, 1]
    assert result['size'].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(StrategyParams()) == 189


if __name__ == '__main__':
    _self_check()
