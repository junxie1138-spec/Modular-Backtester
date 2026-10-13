from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DisplacementShiftParams:
    ma_window: int = 20
    recent_window: int = 4
    reference_window: int = 12
    shift_threshold: float = 1.2
    recovery_ceiling: float = -0.25
    min_size: float = 0.15
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DisplacementShiftParams]):
    strategy_id = 'gen_local-w2_1791334955'

    @classmethod
    def params_type(cls):
        return DisplacementShiftParams

    @staticmethod
    def warmup_bars(params: DisplacementShiftParams) -> int:
        return int(params.ma_window + params.recent_window + params.reference_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: DisplacementShiftParams) -> pd.DataFrame:
        if min(params.ma_window, params.recent_window, params.reference_window) < 2:
            raise ValueError('All windows must be at least two bars.')
        if not np.isfinite(params.shift_threshold) or params.shift_threshold <= 0:
            raise ValueError('shift_threshold must be finite and positive.')
        if not np.isfinite(params.recovery_ceiling):
            raise ValueError('recovery_ceiling must be finite.')
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError('Sizes must satisfy 0 < min_size <= max_size <= 1.')
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        out['ma'] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        out['price_std'] = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out['z'] = (close - out['ma']) / out['price_std'].replace(0.0, np.nan)
        out['recent_z'] = out['z'].rolling(params.recent_window, min_periods=params.recent_window).mean()
        out['reference_z'] = out['z'].shift(params.recent_window).rolling(params.reference_window, min_periods=params.reference_window).mean()
        out['z_step'] = out['z'].diff().abs()
        out['prior_noise'] = out['z_step'].shift(params.recent_window).rolling(params.reference_window, min_periods=params.reference_window).mean()
        out['shift_score'] = (out['recent_z'] - out['reference_z']) / out['prior_noise'].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DisplacementShiftParams) -> SignalFrame:
        score = indicators['shift_score'].replace([np.inf, -np.inf], np.nan)
        z = indicators['z'].replace([np.inf, -np.inf], np.nan)
        entry = (score > params.shift_threshold) & (z < params.recovery_ceiling)
        df = pd.DataFrame(index=data.index)
        # The entry predicate's reversal is the sole exit; no time or price stop.
        df['signal'] = entry.fillna(False).astype(int)
        strength_size = (params.max_size * score / (2.0 * params.shift_threshold)).clip(params.min_size, params.max_size)
        df['size'] = strength_size.shift(1).fillna(params.min_size).astype(float)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    params = DisplacementShiftParams()
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        'z': [-1.0] * 6,
        'shift_score': np.array([np.nan, 1.5, 2.0, 0.5, 1.1, -1.0]) * params.shift_threshold,
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, 1]
    assert result['size'].between(params.min_size, params.max_size).all()
    assert np.isclose(result['size'].iloc[2], 0.75 * params.max_size)
    assert np.isclose(result['size'].iloc[3], params.max_size)
    flat = pd.DataFrame({name: np.ones(80) for name in ['open', 'high', 'low', 'close', 'volume']}, index=pd.date_range('2020-01-01', periods=80))
    flat_result = GeneratedStrategy.generate_signals(flat, GeneratedStrategy.indicators(flat, params), None, params).data
    assert flat_result['signal'].eq(0).all()
    assert np.isfinite(flat_result['size']).all()


if __name__ == '__main__':
    _self_check()
