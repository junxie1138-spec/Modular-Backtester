from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    lookback: int = 20
    threshold: float = 0.20

    def __post_init__(self):
        if not isinstance(self.lookback, int) or isinstance(self.lookback, bool) or self.lookback < 3:
            raise ValueError('lookback must be an integer >= 3')
        if not 0.0 < self.threshold < 1.0:
            raise ValueError('threshold must lie strictly between zero and one')


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = 'gen_local_1791206694'

    @classmethod
    def params_type(cls) -> type[MomentumParams]:
        return MomentumParams

    def warmup_bars(self, params: MomentumParams) -> int:
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['return'] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        window = out['return'].rolling(params.lookback, min_periods=params.lookback)
        out['return_sum'] = window.sum()
        out['best_return'] = window.max()
        out['worst_return'] = window.min()
        out['absolute_sum'] = out['return'].abs().rolling(
            params.lookback, min_periods=params.lookback
        ).sum()
        out['trimmed_drift'] = out['return_sum'] - out['best_return'] - out['worst_return']
        out['trimmed_activity'] = (
            out['absolute_sum'] - out['best_return'].abs() - out['worst_return'].abs()
        ).clip(lower=0.0)
        denominator = out['trimmed_activity'].where(out['trimmed_activity'] > 1e-12)
        out['score'] = (out['trimmed_drift'] / denominator).clip(-1.0, 1.0)
        out.loc[out['trimmed_activity'].le(1e-12), 'score'] = 0.0
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MomentumParams,
    ) -> SignalFrame:
        score = indicators['score'].reindex(data.index)
        decisions = pd.Series(np.nan, index=data.index, dtype=float)
        decisions.loc[score.ge(params.threshold)] = 1.0
        decisions.loc[score.le(-params.threshold)] = 0.0
        df = pd.DataFrame(index=data.index)
        df['signal'] = decisions.ffill().fillna(0).astype(int)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    strategy = GeneratedStrategy()
    params = MomentumParams(lookback=5, threshold=0.20)
    index = pd.date_range('2020-01-01', periods=6, freq='D')
    returns = np.array([0.0, 0.01, 0.01, 0.01, 0.40, -0.02])
    close = 100.0 * np.exp(np.cumsum(returns))
    data = pd.DataFrame(
        {'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1000.0},
        index=index,
    )
    derived = strategy.indicators(data, params)
    assert derived['score'].iloc[:5].isna().all()
    assert np.isclose(derived['score'].iloc[-1], 1.0)
    assert strategy.warmup_bars(params) == 6
    scores = pd.DataFrame({'score': [np.nan, 0.3, 0.1, -0.1, -0.3, 0.3]}, index=index)
    result = strategy.generate_signals(data, scores, None, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
