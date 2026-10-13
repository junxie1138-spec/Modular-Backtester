from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    noise_window: int = 20
    percentile_window: int = 126
    gap_percentile: float = 0.85
    volatility_window: int = 20
    atr_window: int = 14
    target_volatility: float = 0.12
    trailing_atr: float = 1.0
    holding_bars: int = 2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = 'gen_local-w2_1790987600'

    @classmethod
    def params_type(cls) -> type[GapParams]:
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return int(max(
            params.noise_window + params.percentile_window + 2,
            params.volatility_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data['close'].where(data['close'] > 0)
        opening = data['open'].where(data['open'] > 0)
        previous_close = close.shift(1)
        out['gap'] = opening / previous_close - 1.0
        out['intraday'] = close / opening - 1.0
        out['noise'] = out['intraday'].abs().rolling(
            params.noise_window, min_periods=params.noise_window
        ).median().shift(1)
        out['gap_score'] = out['gap'].abs() / out['noise'].where(out['noise'] > 0)
        out['gap_threshold'] = out['gap_score'].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.gap_percentile).shift(1)
        out['return'] = close.pct_change(fill_method=None)
        out['realized_volatility'] = out['return'].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        out['true_range'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - previous_close).abs(),
            (data['low'] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out['atr'] = out['true_range'].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out['target_size'] = (
            params.target_volatility /
            out['realized_volatility'].where(out['realized_volatility'] > 0)
        ).clip(lower=0.001, upper=params.max_size)
        out['candidate'] = 0
        eligible = (
            (out['gap_score'] > out['gap_threshold']) &
            (out['gap'] * out['intraday'] > 0) &
            (out['atr'] > 0) &
            out['target_size'].notna()
        )
        out.loc[eligible & (out['gap'] > 0), 'candidate'] = 1
        out.loc[eligible & (out['gap'] < 0), 'candidate'] = -1
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.001, dtype=float)
        closes = data['close'].to_numpy(dtype=float)
        atrs = indicators['atr'].to_numpy(dtype=float)
        candidates = indicators['candidate'].fillna(0).to_numpy(dtype=int)
        target_sizes = indicators['target_size'].to_numpy(dtype=float)
        position = 0
        age = 0
        highest_close = np.nan
        trailing_stop = -np.inf
        entry_size = 0.001
        holding_limit = max(1, min(2, int(params.holding_bars)))

        for i in range(len(data)):
            if position != 0:
                # The preceding decision entered at this bar's open.
                age += 1
                close = closes[i]
                atr = atrs[i]
                if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                    position = 0
                else:
                    highest_close = close if not np.isfinite(highest_close) else max(highest_close, close)
                    trailing_stop = max(trailing_stop, highest_close - params.trailing_atr * atr)
                    # Apply the requested closing-high drawdown exit to both sides;
                    # on shorts it also realizes a sufficiently large downward move.
                    if close <= trailing_stop or age >= holding_limit:
                        position = 0
                if position == 0:
                    age = 0
                    highest_close = np.nan
                    trailing_stop = -np.inf
                    # Keep the exit decision flat; do not reenter on this close.
                    continue
            elif candidates[i] != 0 and np.isfinite(target_sizes[i]):
                position = int(candidates[i])
                entry_size = float(target_sizes[i])
                age = 0
                highest_close = np.nan
                trailing_stop = -np.inf

            signals[i] = position
            if position != 0:
                sizes[i] = entry_size

        df['signal'] = signals
        df['size'] = sizes
        df['signal'] = df['signal'].shift(1).fillna(0).astype(int)
        df['size'] = df['size'].shift(1).fillna(0.001).astype(float)
        return SignalFrame(data=df, signal_column='signal', size_column='size')
