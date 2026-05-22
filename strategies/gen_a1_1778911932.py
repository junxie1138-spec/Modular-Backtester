from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RisingTideParams:
    env_lookback: int = 20
    atr_len: int = 14
    slope_len: int = 5
    pct_window: int = 120
    entry_pctile: float = 0.80
    min_tide_rate: float = 0.0


class GeneratedStrategy(BaseStrategy[RisingTideParams]):
    """Trend-strength long-only: enter when envelope-midpoint ascent rate times
    close-within-envelope position ranks in its own top rolling percentile.
    Exit only when that entry condition flips (signal-reversal exit).
    """

    strategy_id = "gen_a1_1778911932"

    @classmethod
    def params_type(cls) -> type[RisingTideParams]:
        return RisingTideParams

    @staticmethod
    def warmup_bars(params: RisingTideParams) -> int:
        # tide_rate chains env_lookback -> slope_len shift; the percentile rank
        # then needs pct_window observations of trend_strength on top of that.
        return int(params.env_lookback + params.slope_len + params.pct_window + 5)

    def indicators(self, data: pd.DataFrame, params: RisingTideParams) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_len, min_periods=1).mean()

        # High-low range envelope: rolling ceiling and floor.
        hwm = high.rolling(params.env_lookback, min_periods=1).max()
        lwm = low.rolling(params.env_lookback, min_periods=1).min()
        width = hwm - lwm
        mid = (hwm + lwm) / 2.0

        # Tidal ascent rate: how fast the envelope midpoint drifts up, in
        # ATR units per bar. NaN-safe division (avoid divide-by-zero ATR).
        safe_atr = atr.replace(0.0, np.nan)
        tide_rate = (mid - mid.shift(params.slope_len)) / (
            float(params.slope_len) * safe_atr
        )

        # Position of the close within the high-low envelope, 0..1.
        safe_width = width.replace(0.0, np.nan)
        range_pos = ((close - lwm) / safe_width).clip(0.0, 1.0)

        # Composite trend-strength: rising tide carrying price high in it.
        trend_strength = tide_rate * range_pos

        # Adaptive percentile threshold (the twist): rank the latest score
        # within its own trailing window rather than against a fixed level.
        p = int(params.pct_window)
        ts_pctile = trend_strength.rolling(
            p, min_periods=max(20, p // 3)
        ).rank(pct=True)

        out = pd.DataFrame(index=data.index)
        out["atr"] = atr
        out["hwm"] = hwm
        out["lwm"] = lwm
        out["mid"] = mid
        out["tide_rate"] = tide_rate.fillna(0.0)
        out["range_pos"] = range_pos.fillna(0.0)
        out["trend_strength"] = trend_strength.fillna(0.0)
        out["ts_pctile"] = ts_pctile.fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RisingTideParams,
    ) -> SignalFrame:
        rank = indicators["ts_pctile"].fillna(0.0)
        tide = indicators["tide_rate"].fillna(0.0)

        # Entry condition: trend-strength rank in the top percentile AND the
        # tide (envelope midpoint) is genuinely rising.
        entry = (rank >= float(params.entry_pctile)) & (
            tide > float(params.min_tide_rate)
        )

        # Signal-reversal exit: stay long while the entry condition holds,
        # drive the signal to 0 the moment it flips false. No stop, no timer.
        raw = entry.fillna(False).astype(int)

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
