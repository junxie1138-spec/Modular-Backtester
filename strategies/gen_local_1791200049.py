from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    signature_bars: int = 4
    area_threshold: float = 0.20
    volatility_bars: int = 20
    target_volatility: float = 0.12

    def __post_init__(self):
        if not isinstance(self.signature_bars, int) or self.signature_bars < 2:
            raise ValueError('signature_bars must be an integer >= 2')
        if not isinstance(self.volatility_bars, int) or self.volatility_bars < 2:
            raise ValueError('volatility_bars must be an integer >= 2')
        if not np.isfinite(self.area_threshold) or not 0 < self.area_threshold < 1:
            raise ValueError('area_threshold must lie strictly between zero and one')
        if not np.isfinite(self.target_volatility) or self.target_volatility <= 0:
            raise ValueError('target_volatility must be finite and positive')


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = 'gen_local_1791200049'

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return max(params.signature_bars, params.volatility_bars) + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float)
        close = close.where((close > 0) & np.isfinite(close))
        out['log_return'] = np.log(close).diff()
        n = params.signature_bars
        out['return_budget'] = out['log_return'].abs().rolling(n, min_periods=n).sum()
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric second-level signature for (time, cumulative log price).
        # Each daily segment has time increment one and price increment r.
        for lag in range(n):
            area += 0.5 * (n - 1 - 2 * lag) * out['log_return'].shift(lag)
        out['signature_area'] = area
        denominator = ((n - 1) * out['return_budget']).replace(0.0, np.nan)
        out['area_score'] = (2.0 * area / denominator).replace([np.inf, -np.inf], np.nan)
        out['annual_volatility'] = out['log_return'].rolling(
            params.volatility_bars, min_periods=params.volatility_bars
        ).std(ddof=0) * np.sqrt(252.0)
        out['target_size'] = (
            params.target_volatility / out['annual_volatility'].clip(lower=1e-6)
        ).clip(lower=0.01, upper=0.95)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        score = indicators['area_score']
        ready = np.isfinite(score) & np.isfinite(indicators['annual_volatility'])
        direction = np.select(
            [ready & (score >= params.area_threshold),
             ready & (score <= -params.area_threshold)],
            [1, -1], default=0,
        ).astype(int)
        position = 0
        positions = np.zeros(len(data), dtype=int)
        for i, candidate in enumerate(direction):
            if position != 0 and candidate == -position:
                position = 0
            elif position == 0 and candidate != 0:
                position = int(candidate)
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = positions
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = indicators['target_size'].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = SignatureParams()
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': np.ones(8)}, index=index)
    ind = pd.DataFrame({
        'area_score': [0.5, 0.0, -0.5, -0.5, 0.0, 0.5, 0.5, 0.0],
        'annual_volatility': np.full(8, 0.20),
        'target_size': np.linspace(0.2, 0.9, 8),
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    mirrored = ind.copy()
    mirrored['area_score'] = -mirrored['area_score']
    opposite = GeneratedStrategy.generate_signals(None, data, mirrored, None, params).data
    assert opposite['signal'].equals(-result['signal'])
    assert np.allclose(result['size'].iloc[1:], ind['target_size'].iloc[:-1])
    assert (result['size'] > 0).all()
    prices = pd.DataFrame({'close': np.exp(np.cumsum([0.0, 0.01, 0.02, 0.03, 0.04]))})
    area = GeneratedStrategy.indicators(None, prices, params)
    assert np.isclose(area['area_score'].iloc[-1], 1.0 / 3.0)
    assert area['area_score'].iloc[:4].isna().all()
    assert GeneratedStrategy.warmup_bars(None, params) == 21


if __name__ == '__main__':
    _self_check()
