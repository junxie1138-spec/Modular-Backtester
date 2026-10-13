from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionParams:
    ma_window: int = 20
    overlap_window: int = 8
    percentile_window: int = 126
    extreme_quantile: float = 0.85
    overlap_quantile: float = 0.50
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[AuctionParams]):
    strategy_id = 'gen_local-w2_1791228777'

    @classmethod
    def params_type(cls) -> type[AuctionParams]:
        return AuctionParams

    def warmup_bars(self, params: AuctionParams) -> int:
        if min(params.ma_window, params.overlap_window, params.percentile_window) < 2:
            raise ValueError('Lookbacks must be at least two bars.')
        if not 0.0 < params.extreme_quantile < 1.0:
            raise ValueError('extreme_quantile must be between zero and one.')
        if not 0.0 < params.overlap_quantile < 1.0:
            raise ValueError('overlap_quantile must be between zero and one.')
        if params.hold_bars not in (1, 2):
            raise ValueError('hold_bars must be one or two.')
        return int(max(params.ma_window, params.overlap_window + 1) + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: AuctionParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data['close']
        out['ma'] = close.rolling(params.ma_window).mean()
        out['std'] = close.rolling(params.ma_window).std(ddof=0)
        out['z'] = ((close - out['ma']) / out['std'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out['abs_z'] = out['z'].abs()
        out['z_threshold'] = out['abs_z'].shift(1).rolling(params.percentile_window).quantile(params.extreme_quantile)
        out['range_width'] = (data['high'] - data['low']).clip(lower=0.0)
        out['shared_width'] = (np.minimum(data['high'], data['high'].shift(1)) - np.maximum(data['low'], data['low'].shift(1))).clip(lower=0.0)
        out['smaller_width'] = np.minimum(out['range_width'], out['range_width'].shift(1))
        out['overlap'] = (out['shared_width'] / out['smaller_width'].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
        out['overlap_mean'] = out['overlap'].rolling(params.overlap_window).mean()
        out['overlap_threshold'] = out['overlap_mean'].shift(1).rolling(params.percentile_window).quantile(params.overlap_quantile)
        return out

    @staticmethod
    def _fixed_positions(entries: np.ndarray, hold_bars: int) -> np.ndarray:
        signals = np.zeros(len(entries), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position:
                if i >= exit_bar:
                    position = 0
            elif entry:
                position = int(entry)
                exit_bar = i + hold_bars
            signals[i] = position
        return signals

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AuctionParams) -> SignalFrame:
        self.warmup_bars(params)
        valid = indicators[['z', 'z_threshold', 'overlap_mean', 'overlap_threshold']].notna().all(axis=1)
        extreme = valid & indicators['abs_z'].gt(indicators['z_threshold'])
        fresh = extreme & ~extreme.shift(1, fill_value=False)
        recycling = indicators['overlap_mean'].ge(indicators['overlap_threshold'])
        direction = np.sign(indicators['z'].fillna(0.0)).astype(int)
        entries = np.where(fresh, direction * np.where(recycling, -1, 1), 0).astype(int)
        df = pd.DataFrame(index=data.index)
        df['signal'] = self._fixed_positions(entries, params.hold_bars)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = 0.95
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    entries = np.array([1, -1, -1, 0, 1, -1, 0], dtype=int)
    for hold, expected in ((1, [0, 1, 0, -1, 0, 1, 0]), (2, [0, 1, 1, 0, 0, 1, 1])):
        raw = GeneratedStrategy._fixed_positions(entries, hold)
        shifted = pd.Series(raw).shift(1).fillna(0).astype(int)
        assert shifted.tolist() == expected
