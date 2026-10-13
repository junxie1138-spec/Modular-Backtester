from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownRecyclingParams:
    peak_window: int = 40
    recycling_window: int = 12
    volatility_window: int = 20
    atr_window: int = 14
    minimum_depth: float = 0.02
    maximum_depth: float = 0.12
    recycling_threshold: float = 0.45
    target_volatility: float = 0.14
    breakeven_gain: float = 0.015
    initial_stop_atr: float = 2.5
    trailing_atr: float = 2.0
    max_hold_bars: int = 9
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[DrawdownRecyclingParams]):
    strategy_id = 'gen_local_1790987539'

    @classmethod
    def params_type(cls) -> type[DrawdownRecyclingParams]:
        return DrawdownRecyclingParams

    def warmup_bars(self, params: DrawdownRecyclingParams) -> int:
        return int(max(2, params.peak_window, params.recycling_window + 1,
                       params.volatility_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownRecyclingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['previous_close'] = close.shift(1)
        out['log_return'] = np.log(close / out['previous_close'])
        peak_window = max(2, int(params.peak_window))
        recycling_window = max(2, int(params.recycling_window))
        volatility_window = max(2, int(params.volatility_window))
        atr_window = max(2, int(params.atr_window))
        out['peak'] = close.rolling(peak_window, min_periods=peak_window).max()
        out['drawdown_depth'] = np.log(out['peak'] / close).clip(lower=0.0)
        out['negative_travel'] = (-out['log_return']).clip(lower=0.0)
        out['downside_budget'] = out['negative_travel'].rolling(
            recycling_window, min_periods=recycling_window).sum()
        out['recycling'] = (1.0 - out['drawdown_depth'] /
                            out['downside_budget'].where(out['downside_budget'] > 0)).clip(0.0, 1.0)
        out['annual_volatility'] = out['log_return'].rolling(
            volatility_window, min_periods=volatility_window).std(ddof=0) * np.sqrt(252.0)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - out['previous_close']).abs(),
            (data['low'] - out['previous_close']).abs()
        ], axis=1).max(axis=1).where(out['previous_close'].notna())
        out['atr'] = out['true_range'].rolling(atr_window, min_periods=atr_window).mean()
        depth_scale = (out['drawdown_depth'] /
                       max(2.0 * params.minimum_depth, 1e-6)).clip(0.0, 1.0)
        recycling_scale = ((out['recycling'] - params.recycling_threshold) /
                           max(1.0 - params.recycling_threshold, 1e-6)).clip(0.0, 1.0)
        out['signal_strength'] = depth_scale * recycling_scale
        out['target_size'] = (params.target_volatility /
                              out['annual_volatility'].clip(lower=0.01) *
                              out['signal_strength']).clip(
                                  lower=0.01, upper=max(0.01, min(params.max_size, 1.0)))
        required = ['drawdown_depth', 'recycling', 'annual_volatility',
                    'atr', 'signal_strength', 'target_size']
        out['ready'] = np.isfinite(out[required]).all(axis=1) & (out['atr'] > 0)
        out['eligible'] = (
            out['ready'] &
            out['drawdown_depth'].between(params.minimum_depth, params.maximum_depth) &
            (out['recycling'] > params.recycling_threshold) &
            (close > out['previous_close'])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: DrawdownRecyclingParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.ones(count, dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        lows = data['low'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        eligible = indicators['eligible'].to_numpy(dtype=bool)
        target_sizes = indicators['target_size'].to_numpy(dtype=float)
        held = False
        pending_entry = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        entry_atr = 0.0
        held_size = 1.0
        age = 0

        for i in range(count):
            if held:
                if pending_entry:
                    entry_price = opens[i]
                    pending_entry = False
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        held = False
                        continue
                    stop = max(0.0, entry_price - params.initial_stop_atr * entry_atr)
                age += 1
                valid_bar = np.isfinite(lows[i]) and np.isfinite(closes[i]) and closes[i] > 0
                if (not valid_bar or lows[i] <= stop or
                        age >= max(1, int(params.max_hold_bars))):
                    held = False
                    armed = False
                    continue
                if closes[i] >= entry_price * (1.0 + max(0.0, params.breakeven_gain)):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, closes[i] - params.trailing_atr * atrs[i])
                signals[i] = 1
                sizes[i] = held_size
            elif eligible[i] and i >= self.warmup_bars(params) - 1:
                held = True
                pending_entry = True
                armed = False
                age = 0
                entry_atr = atrs[i]
                held_size = float(target_sizes[i])
                signals[i] = 1
                sizes[i] = held_size

        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
