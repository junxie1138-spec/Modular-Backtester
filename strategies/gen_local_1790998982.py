from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionFadeParams:
    rank_window: int = 40
    tail_fraction: float = 0.15
    sweep_fraction: float = 0.12
    profit_target: float = 0.006
    hold_bars: int = 2
    minimum_size: float = 0.25
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[AuctionFadeParams]):
    strategy_id = 'gen_local_1790998982'

    @classmethod
    def params_type(cls) -> type[AuctionFadeParams]:
        return AuctionFadeParams

    def warmup_bars(self, params: AuctionFadeParams) -> int:
        return max(2, int(params.rank_window))

    def indicators(self, data: pd.DataFrame, params: AuctionFadeParams) -> pd.DataFrame:
        window = max(2, int(params.rank_window))
        result = pd.DataFrame(index=data.index)
        valid_prices = (
            data[['open', 'high', 'low', 'close']].notna().all(axis=1)
            & np.isfinite(data[['open', 'high', 'low', 'close']]).all(axis=1)
            & (data[['open', 'high', 'low', 'close']] > 0).all(axis=1)
            & (data['high'] >= data[['open', 'close']].max(axis=1))
            & (data['low'] <= data[['open', 'close']].min(axis=1))
        )
        result['range'] = (data['high'] - data['low']).where(valid_prices).where(lambda s: s > 0)
        result['impulse'] = ((data['close'] - data['open']) / data['open'].where(data['open'] > 0)).where(valid_prices)
        result['upper_sweep'] = ((data['high'] - data['open']) / result['range']).clip(0.0, 1.0)
        result['lower_sweep'] = ((data['open'] - data['low']) / result['range']).clip(0.0, 1.0)
        result['impulse_rank'] = result['impulse'].rolling(window, min_periods=window).rank(pct=True)
        result['ready'] = result[['range', 'impulse', 'upper_sweep', 'lower_sweep', 'impulse_rank']].notna().all(axis=1)
        result = result.replace([np.inf, -np.inf], np.nan)
        result['ready'] = result['ready'].fillna(False).astype(bool)
        for column in ['range', 'impulse', 'upper_sweep', 'lower_sweep', 'impulse_rank']:
            result[column] = result[column].fillna(0.0)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AuctionFadeParams) -> SignalFrame:
        tail = float(np.clip(params.tail_fraction, 0.01, 0.45))
        sweep = float(np.clip(params.sweep_fraction, 0.0, 0.49))
        target = max(0.0001, float(params.profit_target))
        holding_limit = min(2, max(1, int(params.hold_bars)))
        minimum_size = float(np.clip(params.minimum_size, 0.01, 1.0))
        maximum_size = float(np.clip(params.maximum_size, minimum_size, 1.0))
        rank = indicators['impulse_rank']
        eligible = (
            indicators['ready']
            & (indicators['upper_sweep'] >= sweep)
            & (indicators['lower_sweep'] >= sweep)
        )
        long_entry = eligible & (rank <= tail) & (indicators['impulse'] < 0)
        short_entry = eligible & (rank >= 1.0 - tail) & (indicators['impulse'] > 0)
        candidates = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        extremity = ((rank - 0.5).abs() - (0.5 - tail)).div(tail).clip(0.0, 1.0)
        entry_sizes = (minimum_size + (maximum_size - minimum_size) * extremity).to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), minimum_size, dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        active_size = minimum_size

        for i in range(len(data)):
            if position != 0:
                # The preceding decision fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0.0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and valid_close else -np.inf
                expired = i - entry_bar + 1 >= holding_limit
                if not valid_entry or gain >= target or expired:
                    position = 0
                    entry_price = np.nan
                    active_size = minimum_size
                signals[i] = position
                sizes[i] = active_size
                # An exit decision remains flat; no same-close re-entry.
                continue

            if candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                active_size = float(entry_sizes[i])
                signals[i] = position
                sizes[i] = active_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
