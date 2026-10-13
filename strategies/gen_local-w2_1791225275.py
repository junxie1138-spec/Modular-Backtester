from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringParams:
    box_window: int = 5
    rank_window: int = 60
    squeeze_quantile: float = 0.2
    atr_window: int = 14
    trail_atr: float = 1.5
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[SpringParams]):
    strategy_id = 'gen_local-w2_1791225275'

    @classmethod
    def params_type(cls) -> type[SpringParams]:
        return SpringParams

    def warmup_bars(self, params: SpringParams) -> int:
        return params.box_window + params.rank_window

    def indicators(self, data: pd.DataFrame, params: SpringParams) -> pd.DataFrame:
        high, low, close = data['high'], data['low'], data['close']
        width = (high.rolling(params.box_window).max() - low.rolling(params.box_window).min()) / close.replace(0, np.nan)
        cutoff = width.rolling(params.rank_window).quantile(params.squeeze_quantile)
        previous_close = close.shift(1)
        true_range = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        return pd.DataFrame({
            'width': width.fillna(0.0),
            'cutoff': cutoff.fillna(0.0),
            'squeeze': width.le(cutoff).fillna(False),
            'box_high': high.shift(1).rolling(params.box_window).max().fillna(0.0),
            'box_low': low.shift(1).rolling(params.box_window).min().fillna(0.0),
            'atr': atr.fillna(0.0),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SpringParams) -> SignalFrame:
        compressed = indicators['squeeze'].shift(1, fill_value=False)
        long_entry = compressed & (data['low'] < indicators['box_low']) & (data['close'] > indicators['box_low']) & (data['close'] > data['open'])
        short_entry = compressed & (data['high'] > indicators['box_high']) & (data['close'] < indicators['box_high']) & (data['close'] < data['open'])
        signals = np.zeros(len(data), dtype=int)
        position = age = 0
        peak = trough = stop = 0.0
        for i in range(len(data)):
            close = float(data['close'].iat[i])
            atr = float(indicators['atr'].iat[i])
            if position == 1:
                age += 1
                peak = max(peak, close)
                stop = max(stop, peak - params.trail_atr * atr)
                if close <= stop or age >= params.max_hold:
                    position = 0
            elif position == -1:
                age += 1
                trough = min(trough, close)
                stop = min(stop, trough + params.trail_atr * atr)
                if close >= stop or age >= params.max_hold:
                    position = 0
            elif atr > 0 and long_entry.iat[i]:
                position, age = 1, 0
                peak = close
                stop = close - params.trail_atr * atr
            elif atr > 0 and short_entry.iat[i]:
                position, age = -1, 0
                trough = close
                stop = close + params.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame({'signal': signals, 'size': 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
