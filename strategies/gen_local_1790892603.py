from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 63
    pressure_window: int = 5
    breakout_percentile: float = 0.95
    rejection_percentile: float = 0.30
    holding_bars: int = 18

    def __post_init__(self):
        if self.rank_window < 2 or self.pressure_window < 1 or self.holding_bars < 1:
            raise ValueError('Lookbacks and holding period must be positive; rank_window must exceed one.')
        if not 0.5 < self.breakout_percentile <= 1.0:
            raise ValueError('breakout_percentile must be in (0.5, 1].')
        if not 0.0 < self.rejection_percentile < 0.5:
            raise ValueError('rejection_percentile must be in (0, 0.5).')


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = 'gen_local_1790892603'

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.rank_window + 1, params.rank_window + 2 * params.pressure_window - 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        candle_range = (data['high'] - data['low']).where(data['high'] > data['low'])
        upper_wick = data['high'] - np.maximum(data['open'], data['close'])
        out['rejection_fraction'] = (upper_wick / candle_range).clip(0.0, 1.0)
        out.loc[data['high'].eq(data['low']) & data['close'].notna(), 'rejection_fraction'] = 0.0
        volume = data['volume'].where(data['volume'] >= 0)
        out['rejection_volume'] = out['rejection_fraction'] * volume
        out['window_volume'] = volume.rolling(params.pressure_window, min_periods=params.pressure_window).sum()
        out['rejection_pressure'] = out['rejection_volume'].rolling(params.pressure_window, min_periods=params.pressure_window).sum() / out['window_volume'].where(out['window_volume'] > 0)
        out['rejection_rank'] = out['rejection_pressure'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['prior_rejection_rank'] = out['rejection_rank'].shift(params.pressure_window)
        out['price_rank'] = data['close'].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out['previous_price_rank'] = out['price_rank'].shift(1)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        eligible = (
            indicators['price_rank'].ge(params.breakout_percentile)
            & indicators['previous_price_rank'].lt(params.breakout_percentile)
            & indicators['rejection_rank'].le(params.rejection_percentile)
            & indicators['prior_rejection_rank'].ge(1.0 - params.rejection_percentile)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(eligible):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Reserve the expiry bar for liquidation, even if another entry qualifies.
                    exit_at = -1
            elif enter:
                signal[i] = 1
                exit_at = i + params.holding_bars
        df = pd.DataFrame({'signal': signal, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check():
    params = Params()
    n = params.holding_bars
    index = pd.date_range('2020-01-01', periods=2 * n + 6, freq='B')
    data = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0}, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol='SPY', timeframe='1d', warmup_bars=strategy.warmup_bars(params))
    ranks = pd.Series(0.5, index=index)
    ranks.iloc[[1, n + 1, n + 3]] = 1.0
    indicators = pd.DataFrame({'price_rank': ranks, 'previous_price_rank': ranks.shift(1), 'rejection_rank': 0.0, 'prior_rejection_rank': 1.0}, index=index)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result['signal'].iloc[:2].eq(0).all()
    assert result['signal'].iloc[2:n + 2].eq(1).all()
    assert result['signal'].iloc[n + 2:n + 4].eq(0).all()
    assert result['signal'].iloc[n + 4:2 * n + 4].eq(1).all()
    assert result['signal'].iloc[2 * n + 4:].eq(0).all()
    assert result['size'].gt(0).all()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert strategy.generate_signals(data, derived, ctx, params).data['signal'].eq(0).all()


if __name__ == '__main__':
    _self_check()
