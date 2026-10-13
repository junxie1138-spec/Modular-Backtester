from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ordinal_window: int = 12
    short_window: int = 5
    baseline_window: int = 40
    entropy_max: float = 0.80
    compression_max: float = 0.75
    alternation_min: float = 0.60
    close_location_min: float = 0.60
    hold_bars: int = 4
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = 'gen_local_1790883782'

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ordinal_window + 2, params.short_window, params.baseline_window))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.ordinal_window, params.short_window, params.baseline_window) < 1:
            raise ValueError('Lookback windows must be positive.')
        out = pd.DataFrame(index=data.index)
        close = data['close']
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        # Exclude tied triples rather than assigning artificial ordinal order.
        out['pattern'] = (4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        alternation = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            probability = out['pattern'].eq(code).astype(float).where(valid).rolling(params.ordinal_window, min_periods=params.ordinal_window).mean()
            out[f'pattern_probability_{code}'] = probability
            entropy = entropy - probability * np.log(probability.clip(lower=1e-12))
            if code in (1, 3, 4, 6):
                alternation = alternation + probability
        out['ordinal_entropy'] = entropy / np.log(6.0)
        out['alternation_share'] = alternation
        daily_range = (data['high'] - data['low']).where(data['high'] > data['low'])
        out['normalized_range'] = daily_range / close.where(close > 0)
        out['short_range'] = out['normalized_range'].rolling(params.short_window, min_periods=params.short_window).mean()
        out['baseline_range'] = out['normalized_range'].rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out['compression'] = out['short_range'] / out['baseline_range'].where(out['baseline_range'] > 0)
        out['close_location'] = (close - data['low']) / daily_range
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if params.hold_bars < 1 or not np.isfinite(params.size) or params.size <= 0:
            raise ValueError('Holding period and size must be positive.')
        qualified = (
            indicators['ordinal_entropy'].le(params.entropy_max)
            & indicators['alternation_share'].ge(params.alternation_min)
            & indicators['compression'].le(params.compression_max)
            & indicators['close_location'].ge(params.close_location_min)
            & data['close'].gt(data['open'])
        ).fillna(False)
        confirmed = (qualified & qualified.shift(1, fill_value=False)).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.hold_bars:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif confirmed[i]:
                entry_bar = i
                signal[i] = 1
        df = pd.DataFrame({'signal': signal, 'size': float(params.size)}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


def _self_check() -> None:
    index = pd.date_range('2020-01-01', periods=12)
    data = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0, 'volume': 1000.0}, index=index)
    indicators = pd.DataFrame({'ordinal_entropy': 0.5, 'alternation_share': 0.8, 'compression': 0.5, 'close_location': 0.8}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    indicators.iloc[2:, :] = np.nan
    result = strategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result['signal'].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert result['size'].gt(0).all()


if __name__ == '__main__':
    _self_check()
