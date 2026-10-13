from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PropagationParams:
    fractional_order: float = 0.4
    fractional_window: int = 32
    reference_window: int = 20
    strength_threshold: float = 0.6
    participation_window: int = 8
    participation_threshold: float = 0.55
    volatility_window: int = 20
    target_volatility: float = 0.12
    volatility_floor: float = 0.04
    profit_target: float = 0.04
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[PropagationParams]):
    strategy_id = 'gen_local_1790890601'

    @classmethod
    def params_type(cls) -> type[PropagationParams]:
        return PropagationParams

    def warmup_bars(self, params: PropagationParams) -> int:
        return int(max(
            params.fractional_window + params.reference_window,
            2 * params.participation_window + 1,
            params.volatility_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: PropagationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).where(data['close'] > 0)
        out['log_close'] = np.log(close)
        out['log_return'] = out['log_close'].diff()
        fractional = out['log_close'].copy()
        weight = 1.0
        # ponytail: finite fractional kernel; increase fractional_window if truncation matters.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * out['log_close'].shift(lag)
        out['fractional_close'] = fractional
        out['fractional_mean'] = fractional.shift(1).rolling(params.reference_window).mean()
        out['fractional_std'] = fractional.shift(1).rolling(params.reference_window).std(ddof=0)
        out['fractional_strength'] = (
            (fractional - out['fractional_mean'])
            / out['fractional_std'].replace(0.0, np.nan)
        )
        out['valid_volume'] = data['volume'].astype(float).where(data['volume'] >= 0)
        out['bullish_volume'] = out['valid_volume'].where(out['log_return'] > 0, 0.0)
        out['bullish_volume'] = out['bullish_volume'].where(out['log_return'].notna())
        out['total_volume'] = out['valid_volume'].rolling(params.participation_window).sum()
        out['rising_volume'] = out['bullish_volume'].rolling(params.participation_window).sum()
        out['participation'] = out['rising_volume'] / out['total_volume'].replace(0.0, np.nan)
        out['previous_participation'] = out['participation'].shift(params.participation_window)
        out['participation_growth'] = out['participation'] - out['previous_participation']
        out['annual_volatility'] = out['log_return'].rolling(params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        out['target_size'] = (
            params.target_volatility
            / out['annual_volatility'].clip(lower=params.volatility_floor)
        ).clip(lower=0.05, upper=0.95)
        out['entry'] = (
            (out['fractional_strength'] > params.strength_threshold)
            & (out['participation'] > params.participation_threshold)
            & (out['participation_growth'] > 0.0)
            & out['target_size'].notna()
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PropagationParams) -> SignalFrame:
        df = pd.DataFrame({'signal': 0, 'size': 0.05}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        entries = indicators['entry'].fillna(False).to_numpy(dtype=bool)
        target_sizes = indicators['target_size'].to_numpy(dtype=float)
        opens = data['open'].to_numpy(dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.05
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                expired = i - entry_bar + 1 >= params.holding_bars
                sizes[i] = held_size
                if invalid_entry or target_hit or expired:
                    entry_bar = -1
                    entry_price = np.nan
                    continue
                signals[i] = 1
            elif entries[i] and np.isfinite(target_sizes[i]) and target_sizes[i] > 0:
                held_size = float(target_sizes[i])
                entry_bar = i + 1
                signals[i] = 1
                sizes[i] = held_size
        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')


if __name__ == '__main__':
    strategy = GeneratedStrategy()
    params = PropagationParams(holding_bars=3)
    index = pd.date_range('2020-01-01', periods=6)
    data = pd.DataFrame({
        'open': 100.0, 'high': 106.0, 'low': 99.0,
        'close': 100.0, 'volume': 1000.0,
    }, index=index)
    indicators = pd.DataFrame({'entry': True, 'target_size': 0.5}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 1, 1, 0, 1]
    data.loc[index[1], 'close'] = 105.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result['signal'].tolist() == [0, 1, 0, 1, 1, 1]
    assert result['size'].gt(0).all()
