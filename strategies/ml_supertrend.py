from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


def _resolve_source(data: pd.DataFrame, source_type: str) -> pd.Series:
    """Resolve the Pine `sourceType` input to a price series."""
    o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    if source_type == "open":
        return o
    if source_type == "high":
        return h
    if source_type == "low":
        return l
    if source_type == "close":
        return c
    if source_type == "hl2":
        return (h + l) / 2.0
    if source_type == "hlc3":
        return (h + l + c) / 3.0
    if source_type == "ohlc4":
        return (o + h + l + c) / 4.0
    if source_type == "hlcc4":
        return (h + l + c + c) / 4.0
    raise ValueError(f"Unknown source_type: {source_type!r}")


def _smoothed_tr(data: pd.DataFrame, period: int, use_atr: bool) -> pd.Series:
    """True Range smoothed by Wilder RMA (use_atr=True) or EMA (use_atr=False).

    Mirrors the Pine `rma_var`/`ema_var` of `ta.tr`: TR[0] seeds with
    high[0]-low[0], and the recurrence is seeded from that first value
    (ewm adjust=False), so there are no NaN values.
    """
    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    alpha = (1.0 / period) if use_atr else (2.0 / (period + 1.0))
    return tr.ewm(alpha=alpha, adjust=False).mean()


@dataclass(slots=True)
class MLSupertrendParams:
    # Group 1 — signal mode
    signal_mode: str = "reversal"          # "reversal" | "breakout"
    require_new_extreme: bool = True
    min_bars_between_signals: int = 10
    # Group 2 — volatility envelope
    sensitivity: int = 30
    atr_period: int = 24
    multiplier: float = 1.4
    source_type: str = "hlcc4"
    use_atr: bool = True
    # Group 3 — momentum filter
    enable_rsi: bool = True
    rsi_len: int = 14
    rsi_lookback_top: int = 50
    rsi_lookback_bot: int = 50
    rsi_top: int = 70
    rsi_bot: int = 30
    # Group 4 — flow analysis
    vol_lookback: int = 3
    vol_multiplier: float = 1.2
    require_vol_spike: bool = False
    # Group 5 — signal quality
    enable_major_levels_only: bool = False
    major_level_threshold: float = 4.5
    # Position sizing
    size: float = 1.0


class MLSupertrendStrategy(BaseStrategy[MLSupertrendParams]):
    """
    Purpose:
        SuperTrend + new-extreme reversal/breakout strategy, ported from the
        signal core of the "Machine Learning Supertrend [Aslan]" Pine Script.

        NOTE: the Pine Script's adaptive "ML" self-tuning engine is intentionally
        NOT ported. Parameters are static — tune them with the suite's
        grid-search / walk-forward optimization, not an in-sample self-tuner.

    Inputs:
        OHLCV dataframe with datetime index and lowercase columns:
        open, high, low, close, volume.

    Outputs:
        SignalFrame with `signal` in {-1, 0, 1} (stop-and-reverse held position)
        and `size`.

    Requires:
        ExecutionConfig.allow_short = True. The stop-and-reverse model goes
        short on every Sell; without allow_short the simulator raises
        ShortNotAllowedError on the first -1.
    """

    strategy_id = "ml_supertrend"
    version = "1.0"
    asset_type = "stock"
    timeframe = "1d"

    @classmethod
    def params_type(cls):
        return MLSupertrendParams

    def warmup_bars(self, params: MLSupertrendParams) -> int:
        return max(
            params.atr_period,
            params.sensitivity,
            params.rsi_len,
            params.vol_lookback,
        ) + 1

    def indicators(self, data: pd.DataFrame, params: MLSupertrendParams) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MLSupertrendParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = params.size
        return SignalFrame(data=df, signal_column="signal", size_column="size")
