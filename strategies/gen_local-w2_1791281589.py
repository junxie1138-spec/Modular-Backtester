from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    lookback: int = 20
    imbalance_threshold: float = 0.15

    def __post_init__(self):
        if self.lookback < 2:
            raise ValueError('lookback must be at least 2')
        if not np.isfinite(self.imbalance_threshold) or self.imbalance_threshold <= 0:
            raise ValueError('imbalance_threshold must be finite and positive')


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = 'gen_local-w2_1791281589'

    @classmethod
    def params_type(cls):
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return max(200, params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1).replace(0.0, np.nan)
        out['gap'] = data['open'].div(out['previous_close']) - 1.0
        out['intraday'] = (data['close'] - data['open']).div(out['previous_close'])
        out['absorption'] = (-np.sign(out['gap']) * out['intraday']).clip(lower=0.0)
        out['residual_gap'] = np.sign(out['gap']) * (out['gap'].abs() - out['absorption']).clip(lower=0.0)
        out['up_residual'] = out['residual_gap'].clip(lower=0.0)
        out['down_residual'] = (-out['residual_gap']).clip(lower=0.0)
        window = params.lookback
        out['gap_volatility'] = out['gap'].rolling(window, min_periods=window).std(ddof=0)
        out['up_volatility'] = out['up_residual'].rolling(window, min_periods=window).std(ddof=0)
        out['down_volatility'] = out['down_residual'].rolling(window, min_periods=window).std(ddof=0)
        denominator = out['gap_volatility'].replace(0.0, np.nan)
        out['imbalance'] = (out['up_volatility'] - out['down_volatility']).div(denominator)
        out['ma200'] = data['close'].rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapParams) -> SignalFrame:
        ready = indicators[['imbalance', 'ma200']].notna().all(axis=1)
        long_entry = ready & indicators['imbalance'].gt(params.imbalance_threshold) & data['close'].gt(indicators['ma200'])
        short_entry = ready & indicators['imbalance'].lt(-params.imbalance_threshold) & data['close'].lt(indicators['ma200'])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(candidates):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                # Opposite entry evidence exits; re-entry requires a later bar.
                position = 0
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    index = pd.date_range('2020-01-01', periods=8)
    data = pd.DataFrame({'close': [100., 101., 99., 99., 99., 101., 101., 101.]}, index=index)
    indicators = pd.DataFrame({'imbalance': [np.nan, .4, 0., -.4, -.4, 0., .4, .4], 'ma200': 100.}, index=index)
    strategy = GeneratedStrategy()
    params = GapParams()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
    assert strategy.warmup_bars(GapParams(lookback=250)) == 251


if __name__ == '__main__':
    _self_check()
