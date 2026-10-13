from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEscapeParams:
    range_window: int = 20
    channel_window: int = 6
    pressure_window: int = 5
    atr_window: int = 14
    expansion_multiple: float = 1.4
    overlap_ceiling: float = 0.35
    pressure_threshold: float = 0.2
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[RangeEscapeParams]):
    strategy_id = "gen_local-w2_1790990820"

    @classmethod
    def params_type(cls) -> type[RangeEscapeParams]:
        return RangeEscapeParams

    def warmup_bars(self, params: RangeEscapeParams) -> int:
        return int(max(params.range_window + 1, params.channel_window + 1,
                       params.pressure_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeEscapeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float).clip(lower=0)
        out["range"] = (high - low).clip(lower=0)
        out["range_baseline"] = out["range"].rolling(params.range_window).mean().shift(1)
        out["range_expansion"] = out["range"] / out["range_baseline"].replace(0, np.nan)
        out["overlap_length"] = (np.minimum(high, high.shift(1)) -
                                   np.maximum(low, low.shift(1))).clip(lower=0)
        out["overlap"] = out["overlap_length"] / np.minimum(
            out["range"], out["range"].shift(1)).replace(0, np.nan)
        out["channel_high"] = high.rolling(params.channel_window).max().shift(1)
        out["channel_low"] = low.rolling(params.channel_window).min().shift(1)
        out["close_location"] = ((2.0 * close - high - low) /
                                   out["range"].replace(0, np.nan)).fillna(0.0).clip(-1, 1)
        out["weighted_pressure"] = volume * out["close_location"]
        out["pressure"] = out["weighted_pressure"].rolling(params.pressure_window).sum() / \
            volume.rolling(params.pressure_window).sum().replace(0, np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            out["range"], (high - previous_close).abs(),
            (low - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeEscapeParams) -> SignalFrame:
        valid = indicators[["range_expansion", "overlap", "channel_high",
                            "channel_low", "pressure", "atr"]].notna().all(axis=1)
        valid &= indicators["atr"].gt(0)
        range_escape = (indicators["range_expansion"].ge(params.expansion_multiple) &
                        indicators["overlap"].le(params.overlap_ceiling))
        long_entry = (valid & range_escape &
                      data["close"].gt(indicators["channel_high"]) &
                      indicators["pressure"].ge(params.pressure_threshold))
        short_entry = (valid & range_escape &
                       data["close"].lt(indicators["channel_low"]) &
                       indicators["pressure"].le(-params.pressure_threshold))
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        direction = 0
        pending_entry = False
        entry = 0.0
        stop = 0.0
        peak = 0.0
        armed = False
        held = 0
        for i in range(len(data)):
            exited = False
            if pending_entry:
                pending_entry = False
                entry = opens[i]
                if not np.isfinite(entry) or entry <= 0:
                    direction = 0
                    exited = True
                else:
                    peak = direction * entry
                    stop = peak - params.trail_atr * atr[i - 1]
                    armed = False
                    held = 0
            if direction != 0:
                held += 1
                # Signed coordinates make the stop ratchet upward for both sides.
                # Check the previously established stop before updating at this close.
                adverse = lows[i] if direction == 1 else -highs[i]
                if np.isfinite(adverse) and adverse <= stop:
                    direction = 0
                    exited = True
                elif held >= params.max_hold_bars:
                    direction = 0
                    exited = True
                elif np.isfinite(closes[i]):
                    signed_close = direction * closes[i]
                    peak = max(peak, signed_close)
                    if signed_close - direction * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, direction * entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atr[i])
                        if signed_close <= stop:
                            direction = 0
                            exited = True
            if direction == 0 and not exited and candidates[i] != 0:
                direction = int(candidates[i])
                pending_entry = True
            raw[i] = direction
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
