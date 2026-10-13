from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalReclamationParams:
    fractional_order: float = 0.4
    fractional_window: int = 64
    percentile_window: int = 126
    tail_percentile: float = 0.2
    residence_window: int = 5
    minimum_tail_visits: int = 2
    volatility_window: int = 20
    volatility_percentile: float = 0.7
    target_volatility: float = 0.12
    maximum_size: float = 0.95
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[FractionalReclamationParams]):
    strategy_id = 'gen_local-w2_1790989989'

    @classmethod
    def params_type(cls) -> type[FractionalReclamationParams]:
        return FractionalReclamationParams

    def warmup_bars(self, params: FractionalReclamationParams) -> int:
        return int(max(
            params.fractional_window + params.percentile_window + params.residence_window + 1,
            params.volatility_window + params.percentile_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: FractionalReclamationParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].astype(float).replace([np.inf, -np.inf], np.nan)
        weights = np.ones(params.fractional_window, dtype=float)
        for k in range(1, params.fractional_window):
            weights[k] = -weights[k - 1] * (params.fractional_order - k + 1.0) / k
        if len(close):
            filtered = np.convolve(close.to_numpy(), weights, mode='full')[:len(close)]
            filtered[:params.fractional_window - 1] = np.nan
        else:
            filtered = np.empty(0, dtype=float)
        out['fractional_close'] = pd.Series(filtered, index=data.index).replace([np.inf, -np.inf], np.nan)
        history = out['fractional_close'].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        out['lower_threshold'] = history.quantile(params.tail_percentile)
        out['median_threshold'] = history.quantile(0.5)
        valid_tail = out['fractional_close'].notna() & out['lower_threshold'].notna()
        out['tail_visit'] = (out['fractional_close'] < out['lower_threshold']).astype(float).where(valid_tail)
        out['prior_tail_visits'] = out['tail_visit'].shift(1).rolling(
            params.residence_window, min_periods=params.residence_window
        ).sum()
        out['return'] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out['volatility'] = out['return'].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        out['volatility_threshold'] = out['volatility'].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.volatility_percentile)
        out['target_size'] = (
            params.target_volatility / out['volatility'].where(out['volatility'] > 0)
        ).clip(lower=0.01, upper=params.maximum_size).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FractionalReclamationParams,
    ) -> SignalFrame:
        fd = indicators['fractional_close']
        median = indicators['median_threshold']
        ready = indicators[[
            'fractional_close', 'median_threshold', 'prior_tail_visits',
            'volatility', 'volatility_threshold', 'target_size',
        ]].notna().all(axis=1)
        reclaim = (fd > median) & (fd.shift(1) <= median.shift(1))
        candidate = (
            ready & reclaim
            & (indicators['prior_tail_visits'] >= params.minimum_tail_visits)
            & (indicators['volatility'] > 0)
            & (indicators['volatility'] <= indicators['volatility_threshold'])
        ).fillna(False).to_numpy(dtype=bool)
        target_sizes = indicators['target_size'].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), float(params.maximum_size), dtype=float)
        exit_at = -1
        entry_size = float(params.maximum_size)
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signals[i] = 1
                    sizes[i] = entry_size
                else:
                    # Force a flat bar at expiry, even if a new entry qualifies.
                    exit_at = -1
            elif candidate[i]:
                signals[i] = 1
                entry_size = float(target_sizes[i])
                sizes[i] = entry_size
                exit_at = i + params.holding_bars
        df = pd.DataFrame({'signal': signals, 'size': sizes}, index=data.index)
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(float(params.maximum_size)).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
