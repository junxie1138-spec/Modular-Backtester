from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityResponseParams:
    atr_window: int = 14
    memory_window: int = 63
    normalization_window: int = 42
    minimum_clustering: float = 0.10
    entry_strength: float = 0.30
    breakeven_gain: float = 0.025
    initial_stop_atr: float = 2.5
    trail_atr: float = 3.0
    holding_bars: int = 18
    minimum_size: float = 0.20
    maximum_size: float = 0.95


class GeneratedStrategy(BaseStrategy[VolatilityResponseParams]):
    strategy_id = 'gen_local-w2_1790985551'

    @classmethod
    def params_type(cls) -> type[VolatilityResponseParams]:
        return VolatilityResponseParams

    @staticmethod
    def warmup_bars(params: VolatilityResponseParams) -> int:
        return int(max(2, params.atr_window) + max(3, params.memory_window, params.normalization_window) + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolatilityResponseParams) -> pd.DataFrame:
        atr_window = max(2, int(params.atr_window))
        memory = max(3, int(params.memory_window))
        normalization = max(3, int(params.normalization_window))
        out = pd.DataFrame(index=data.index)
        out['previous_close'] = data['close'].shift(1)
        out['return'] = data['close'].pct_change(fill_method=None)
        out['absolute_return'] = out['return'].abs()
        out['lag_absolute_return'] = out['absolute_return'].shift(1)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(atr_window, min_periods=atr_window).mean()
        out['atr_fraction'] = out['atr'] / data['close'].where(data['close'] > 0)
        out['volatility_innovation'] = out['atr_fraction'].diff()
        out['innovation_scale'] = out['volatility_innovation'].rolling(
            normalization, min_periods=normalization
        ).std(ddof=0)
        out['standardized_innovation'] = out['volatility_innovation'] / out['innovation_scale'].where(
            out['innovation_scale'] > 1e-12
        )
        out['lag_innovation'] = out['volatility_innovation'].shift(1)
        out['clustering'] = out['absolute_return'].rolling(memory, min_periods=memory).corr(
            out['lag_absolute_return']
        )
        out['response'] = out['return'].rolling(memory, min_periods=memory).corr(
            out['lag_innovation']
        )
        out['strength'] = out['response'] * out['standardized_innovation'].clip(-4.0, 4.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out['ready'] = out[['atr', 'clustering', 'strength']].notna().all(axis=1)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolatilityResponseParams,
    ) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        highs = data['high'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        strengths = indicators['strength'].to_numpy(dtype=float)
        clustering = indicators['clustering'].to_numpy(dtype=float)
        ready = indicators['ready'].to_numpy(dtype=bool)
        minimum_size = max(0.01, min(1.0, float(params.minimum_size)))
        maximum_size = max(minimum_size, min(1.0, float(params.maximum_size)))
        threshold = max(1e-6, float(params.entry_strength))
        initial_multiple = max(0.01, float(params.initial_stop_atr))
        trail_multiple = max(0.01, float(params.trail_atr))
        activation = max(0.0, float(params.breakeven_gain))
        holding_bars = max(1, int(params.holding_bars))
        position = 0
        pending = 0
        pending_size = minimum_size
        position_size = minimum_size
        entry_price = 0.0
        entry_bar = 0
        stop = 0.0
        armed = False

        for i in range(count):
            exited = False
            if pending:
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    position = pending
                    position_size = pending_size
                    entry_price = opens[i]
                    entry_bar = i
                    # Signed profit coordinates make the stop increase for both directions.
                    stop = -initial_multiple * atrs[i - 1]
                    armed = False
                pending = 0

            if position:
                adverse_price = lows[i] if position == 1 else highs[i]
                adverse_profit = position * (adverse_price - entry_price)
                stop_hit = np.isfinite(adverse_profit) and adverse_profit <= stop
                expired = i - entry_bar + 1 >= holding_bars
                if stop_hit or expired:
                    position = 0
                    armed = False
                    exited = True
                elif np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0:
                    close_profit = position * (closes[i] - entry_price)
                    if close_profit / entry_price >= activation:
                        armed = True
                    if armed:
                        # Test the existing stop first; this close's ratchet starts next bar.
                        stop = max(stop, 0.0, close_profit - trail_multiple * atrs[i])

            if position:
                signals[i] = position
                sizes[i] = position_size
            elif not exited and ready[i] and clustering[i] >= params.minimum_clustering and abs(strengths[i]) >= threshold:
                pending = 1 if strengths[i] > 0 else -1
                conviction = float(np.clip(abs(strengths[i]) / (3.0 * threshold), 0.0, 1.0))
                pending_size = minimum_size + (maximum_size - minimum_size) * conviction
                signals[i] = pending
                sizes[i] = pending_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(minimum_size).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
