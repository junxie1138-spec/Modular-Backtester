from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapResponseParams:
    lookback: int = 8
    gap_threshold: float = 1.8
    residual_threshold: float = 0.5

    def __post_init__(self):
        if not 3 <= self.lookback <= 8:
            raise ValueError('lookback must be between 3 and 8')
        if not np.isfinite(self.gap_threshold) or self.gap_threshold <= 0:
            raise ValueError('gap_threshold must be finite and positive')
        if not np.isfinite(self.residual_threshold) or self.residual_threshold <= 0:
            raise ValueError('residual_threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[GapResponseParams]):
    strategy_id = 'gen_local-w2_1791283756'

    @classmethod
    def params_type(cls):
        return GapResponseParams

    def warmup_bars(self, params: GapResponseParams) -> int:
        return int(params.lookback + 2)

    def indicators(self, data: pd.DataFrame, params: GapResponseParams) -> pd.DataFrame:
        n = params.lookback
        opening = data['open'].where(data['open'] > 0)
        closing = data['close'].where(data['close'] > 0)
        gap = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        body = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        # Fit exclusively on prior sessions; current observations are predictions.
        gap_mean = gap.rolling(n, min_periods=n).mean().shift(1)
        body_mean = body.rolling(n, min_periods=n).mean().shift(1)
        gap_var = gap.rolling(n, min_periods=n).var(ddof=0).shift(1)
        body_var = body.rolling(n, min_periods=n).var(ddof=0).shift(1)
        covariance = (gap * body).rolling(n, min_periods=n).mean().shift(1) - gap_mean * body_mean
        gap_var = gap_var.where(gap_var > 1e-12)
        beta = covariance / gap_var
        predicted_body = body_mean + beta * (gap - gap_mean)
        residual_var = (body_var - covariance.pow(2) / gap_var).clip(lower=0)
        residual_std = np.sqrt(residual_var).where(residual_var > 1e-12)
        gap_std = np.sqrt(gap_var)
        out = pd.DataFrame({
            'gap': gap,
            'body': body,
            'gap_mean': gap_mean,
            'body_mean': body_mean,
            'gap_var': gap_var,
            'body_var': body_var,
            'covariance': covariance,
            'beta': beta,
            'predicted_body': predicted_body,
            'residual_var': residual_var,
            'residual_std': residual_std,
            'gap_std': gap_std,
            'gap_z': (gap - gap_mean) / gap_std,
            'residual_z': (body - predicted_body) / residual_std,
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapResponseParams) -> SignalFrame:
        long_entry = (
            (indicators['gap'] > 0)
            & (indicators['gap_z'] >= params.gap_threshold)
            & (indicators['residual_z'] >= params.residual_threshold)
        ).fillna(False)
        short_entry = (
            (indicators['gap'] < 0)
            & (indicators['gap_z'] <= -params.gap_threshold)
            & (indicators['residual_z'] <= -params.residual_threshold)
        ).fillna(False)
        evidence = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(evidence):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Opposite entry evidence exits to flat for this decision bar.
                position = 0
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = GapResponseParams()
    strategy = GeneratedStrategy()
    index = pd.date_range('2020-01-01', periods=7)
    data = pd.DataFrame(index=index)
    evidence = np.array([1, 0, -1, -1, 0, 1, 0])
    indicators = pd.DataFrame({
        'gap': evidence.astype(float),
        'gap_z': evidence * 3.0,
        'residual_z': evidence * 2.0,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result['size'] > 0).all()
    assert strategy.warmup_bars(params) <= 10


if __name__ == '__main__':
    _self_check()
