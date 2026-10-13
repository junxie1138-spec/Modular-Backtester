from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCompressionParams:
    compression_window: int = 10
    reference_window: int = 60
    regime_window: int = 200
    compression_ratio: float = 0.35
    minimum_gap_fraction: float = 0.12
    retained_gap_fraction: float = 0.70
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapCompressionParams]):
    strategy_id = 'gen_local-w2_1790988874'

    @classmethod
    def params_type(cls) -> type[GapCompressionParams]:
        return GapCompressionParams

    def warmup_bars(self, params: GapCompressionParams) -> int:
        return int(max(
            max(1, params.compression_window) + 1,
            max(1, params.reference_window) + 1,
            max(200, params.regime_window),
        ))

    def indicators(self, data: pd.DataFrame, params: GapCompressionParams) -> pd.DataFrame:
        short_window = max(1, int(params.compression_window))
        long_window = max(1, int(params.reference_window))
        regime_window = max(200, int(params.regime_window))
        out = pd.DataFrame(index=data.index)
        out['prior_close'] = data['close'].shift(1)
        out['box_high'] = data['high'].rolling(short_window, min_periods=short_window).max().shift(1)
        out['box_low'] = data['low'].rolling(short_window, min_periods=short_window).min().shift(1)
        out['reference_high'] = data['high'].rolling(long_window, min_periods=long_window).max().shift(1)
        out['reference_low'] = data['low'].rolling(long_window, min_periods=long_window).min().shift(1)
        out['box_width'] = out['box_high'] - out['box_low']
        out['reference_width'] = out['reference_high'] - out['reference_low']
        out['compression'] = out['box_width'].div(out['reference_width'].where(out['reference_width'] > 0))
        out['gap'] = data['open'] - out['prior_close']
        out['gap_fraction'] = out['gap'].abs().div(out['box_width'].where(out['box_width'] > 0))
        out['retention_level'] = out['prior_close'] + params.retained_gap_fraction * out['gap']
        out['regime_ma'] = data['close'].rolling(regime_window, min_periods=regime_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapCompressionParams,
    ) -> SignalFrame:
        valid = indicators.notna().all(axis=1) & data[['open', 'close']].notna().all(axis=1)
        compressed = indicators['compression'].le(params.compression_ratio)
        substantial = indicators['gap_fraction'].ge(params.minimum_gap_fraction)
        eligible = valid & compressed & substantial & indicators['box_width'].gt(0)
        long_entry = (
            eligible
            & indicators['gap'].gt(0)
            & data['open'].gt(indicators['box_high'])
            & data['close'].gt(indicators['box_high'])
            & data['close'].ge(indicators['retention_level'])
            & data['close'].gt(indicators['regime_ma'])
        )
        short_entry = (
            eligible
            & indicators['gap'].lt(0)
            & data['open'].lt(indicators['box_low'])
            & data['close'].lt(indicators['box_low'])
            & data['close'].le(indicators['retention_level'])
            & data['close'].lt(indicators['regime_ma'])
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        hold_bars = max(1, int(params.hold_bars))
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0 and i >= exit_bar:
                position = 0
            elif position == 0 and entries[i] != 0:
                position = int(entries[i])
                exit_bar = i + hold_bars
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df['signal'] = signals
        df['size'] = 0.95
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
