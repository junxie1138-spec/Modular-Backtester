from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ExhaustionParams:
    depth_window: int = 20
    percentile_window: int = 100
    tail_quantile: float = 0.85
    volume_window: int = 20
    profit_target: float = 0.007
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[ExhaustionParams]):
    strategy_id = 'gen_local-w2_1791281972'

    @classmethod
    def params_type(cls):
        return ExhaustionParams

    def warmup_bars(self, params: ExhaustionParams) -> int:
        return max(params.depth_window + params.percentile_window, params.volume_window)

    def indicators(self, data: pd.DataFrame, params: ExhaustionParams) -> pd.DataFrame:
        if min(params.depth_window, params.percentile_window, params.volume_window) < 2:
            raise ValueError('Rolling windows must be at least two bars.')
        if not 0.5 < params.tail_quantile < 1.0:
            raise ValueError('tail_quantile must lie between 0.5 and 1.')
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError('profit_target must be positive and finite.')
        if params.max_hold_bars not in (1, 2):
            raise ValueError('max_hold_bars must be one or two.')
        out = pd.DataFrame(index=data.index)
        out['peak'] = data['close'].rolling(params.depth_window, min_periods=params.depth_window).max()
        out['trough'] = data['close'].rolling(params.depth_window, min_periods=params.depth_window).min()
        out['drawdown'] = 1.0 - data['close'] / out['peak'].where(out['peak'] > 0)
        out['drawup'] = data['close'] / out['trough'].where(out['trough'] > 0) - 1.0
        out['drawdown_threshold'] = out['drawdown'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.tail_quantile)
        out['drawup_threshold'] = out['drawup'].shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.tail_quantile)
        out['volume_median'] = data['volume'].rolling(params.volume_window, min_periods=params.volume_window).median()
        out['lower_wick'] = data[['open', 'close']].min(axis=1, skipna=False) - data['low']
        out['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1, skipna=False)
        out['body'] = data['close'] - data['open']
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1) & data.notna().all(axis=1)
        thin = (data['volume'] > 0) & (data['volume'] < out['volume_median'])
        buy = ready & thin & (out['drawdown'] > out['drawdown_threshold']) & (out['body'] > 0) & (out['lower_wick'] > out['upper_wick'])
        sell = ready & thin & (out['drawup'] > out['drawup_threshold']) & (out['body'] < 0) & (out['upper_wick'] > out['lower_wick'])
        out['entry'] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ExhaustionParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators['entry'].fillna(0).to_numpy(dtype=int)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # The previous close's entry fills at this bar's open.
                if entry_bar < 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        raise ValueError('An entry requires a positive finite opening price.')
                    entry_price = opens[i]
                    entry_bar = i
                gain = position * (closes[i] / entry_price - 1.0)
                held = i - entry_bar + 1
                # Close-confirmed exits fill at the following bar's open.
                if gain >= params.profit_target or held >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
                    entry_bar = -1
            elif entries[i]:
                position = int(entries[i])
                entry_bar = -1
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 0.95}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    index = pd.date_range('2020-01-01', periods=6)
    sample = pd.DataFrame({'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': [100.0, 101.0, 100.0, 100.0, 100.0, 100.0], 'volume': 1000.0}, index=index)
    evidence = pd.DataFrame({'entry': [1, 0, -1, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, sample, evidence, None, ExhaustionParams()).data
    assert result['signal'].tolist() == [0, 1, 0, -1, -1, 0]
    assert result['size'].gt(0).all()
