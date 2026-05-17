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


def _supertrend_trend(
    src: np.ndarray,
    atr: np.ndarray,
    close: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """SuperTrend trend state in {+1, -1}, faithful to Pine `getSupertrend_var`.

    `support` is the band below price, `resistance` the band above. Bands
    ratchet using the *previous* close vs the *previous* band; the trend flips
    using the *current* close vs the *previous* band. Trend starts at +1.

    (The Pine code names these `upper`/`lower` with swapped meanings — see spec
    section 6.2. Behaviour here is identical.)
    """
    src = np.asarray(src, dtype=float)
    atr = np.asarray(atr, dtype=float)
    close = np.asarray(close, dtype=float)
    n = src.shape[0]
    support = src - multiplier * atr
    resistance = src + multiplier * atr
    trend = np.ones(n, dtype=np.int64)
    for i in range(1, n):
        if close[i - 1] > support[i - 1]:
            support[i] = max(support[i], support[i - 1])
        if close[i - 1] < resistance[i - 1]:
            resistance[i] = min(resistance[i], resistance[i - 1])
        if trend[i - 1] == -1 and close[i] > resistance[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close[i] < support[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return trend


def _wilder_rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder RSI (alpha = 1/length), matching Pine `ta.rsi`. Wilder smoothing
    as in `rsi_long_short.py`; the first `length` values are NaN. A window with
    no down moves yields RSI = 100 (the Pine convention) rather than NaN, so the
    RSI filter is not silently disabled during strong rallies."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 with gains present -> rs is +inf -> RSI = 100. The replace()
    # above turned that into NaN; restore 100 explicitly.
    no_loss = (avg_loss == 0.0) & (avg_gain > 0.0)
    return rsi.mask(no_loss, 100.0)


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
        out = pd.DataFrame(index=data.index)
        close = data["close"]

        src = _resolve_source(data, params.source_type)
        atr = _smoothed_tr(data, params.atr_period, params.use_atr)
        out["atr"] = atr
        out["st_trend"] = _supertrend_trend(
            src.to_numpy(), atr.to_numpy(), close.to_numpy(), params.multiplier
        )
        out["rsi"] = _wilder_rsi(close, params.rsi_len)

        # Rolling extremes over the sensitivity window.
        roll_high = data["high"].rolling(params.sensitivity).max()
        roll_low = data["low"].rolling(params.sensitivity).min()
        out["roll_high"] = roll_high
        out["roll_low"] = roll_low

        # Fresh-pivot detection: the rolling extreme changed vs `lookback` bars
        # ago AND close pushed past that prior extreme.
        lookback = max(1, int(round(params.sensitivity / 10.0)))
        prev_high = roll_high.shift(lookback)
        prev_low = roll_low.shift(lookback)
        out["is_new_high"] = (
            roll_high.notna() & prev_high.notna()
            & (roll_high != prev_high) & (close > prev_high)
        )
        out["is_new_low"] = (
            roll_low.notna() & prev_low.notna()
            & (roll_low != prev_low) & (close < prev_low)
        )

        # RSI hot/cold memory: was RSI past the threshold within the lookback?
        if params.enable_rsi:
            cold = out["rsi"] < params.rsi_bot
            hot = out["rsi"] > params.rsi_top
            out["rsi_cold"] = (
                cold.rolling(params.rsi_lookback_bot, min_periods=1).max()
                .fillna(0.0).astype(bool)
            )
            out["rsi_hot"] = (
                hot.rolling(params.rsi_lookback_top, min_periods=1).max()
                .fillna(0.0).astype(bool)
            )
        else:
            out["rsi_cold"] = pd.Series(True, index=data.index)
            out["rsi_hot"] = pd.Series(True, index=data.index)

        # Volume surge.
        vol_avg = data["volume"].rolling(params.vol_lookback).mean()
        out["vol_surge"] = (data["volume"] > params.vol_multiplier * vol_avg).fillna(False)

        # Key-levels filter: only the biggest structural pivots survive.
        if params.enable_major_levels_only:
            depth = atr * params.major_level_threshold
            out["sig_high"] = out["is_new_high"] & ((data["high"] - roll_low) > depth)
            out["sig_low"] = out["is_new_low"] & ((roll_high - data["low"]) > depth)
        else:
            out["sig_high"] = out["is_new_high"]
            out["sig_low"] = out["is_new_low"]

        return out

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
