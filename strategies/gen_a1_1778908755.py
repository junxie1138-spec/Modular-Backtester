from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StandingWaveNodeParams:
    dd_window: int = 40
    rank_window: int = 120
    comp_window: int = 12
    comp_pct: float = 0.30
    low_dd_pct: float = 0.20
    high_dd_pct: float = 0.80
    hold_bars: int = 4
    atr_window: int = 20
    vol_target: float = 0.012
    size_floor: float = 0.5
    size_cap: float = 1.6


def _roll_rank(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of the last value within its window (NaN-safe)."""
    def _rank(x: np.ndarray) -> float:
        last = x[-1]
        if not np.isfinite(last):
            return np.nan
        return float(np.mean(x <= last))

    return s.rolling(window, min_periods=window).apply(_rank, raw=True)


class GeneratedStrategy(BaseStrategy[StandingWaveNodeParams]):
    strategy_id = "gen_a1_1778908755"

    @classmethod
    def params_type(cls) -> type[StandingWaveNodeParams]:
        return StandingWaveNodeParams

    @staticmethod
    def warmup_bars(params: StandingWaveNodeParams) -> int:
        return int(
            max(
                params.dd_window + params.rank_window,
                params.comp_window + params.rank_window,
                params.atr_window,
            )
            + 5
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: StandingWaveNodeParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ind = pd.DataFrame(index=data.index)

        # Drawdown depth relative to the rolling peak close (always <= 0).
        roll_max = close.rolling(params.dd_window, min_periods=params.dd_window).max()
        drawdown = close / roll_max - 1.0
        ind["drawdown"] = drawdown

        # Percentile threshold (the twist): rank drawdown depth within its own history.
        ind["dd_rank"] = _roll_rank(drawdown, params.rank_window)

        # Range compression: smoothed bar range as a fraction of price, then ranked.
        safe_close = close.replace(0.0, np.nan)
        bar_range = (high - low) / safe_close
        range_ma = bar_range.rolling(
            params.comp_window, min_periods=params.comp_window
        ).mean()
        ind["comp_rank"] = _roll_rank(range_ma, params.rank_window)

        # ATR% for volatility-targeted sizing.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        ind["atr_pct"] = atr / safe_close

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StandingWaveNodeParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        comp_rank = indicators["comp_rank"].to_numpy(dtype=float)
        dd_rank = indicators["dd_rank"].to_numpy(dtype=float)
        atr_pct = indicators["atr_pct"].to_numpy(dtype=float)

        # Compression is a symmetric gate; NaN comparisons yield False (no warmup entries).
        compressed = comp_rank < params.comp_pct
        long_entry = compressed & (dd_rank < params.low_dd_pct)
        short_entry = compressed & (dd_rank > params.high_dd_pct)

        raw = np.zeros(n, dtype=np.int64)
        raw[long_entry] = 1
        raw[short_entry] = -1

        # Fixed-bar exit: hold exactly hold_bars bars after entry, no signal-based exit.
        hold = max(1, int(params.hold_bars))
        pos = np.zeros(n, dtype=np.int64)
        i = 0
        while i < n:
            d = raw[i]
            if d != 0:
                end = min(i + hold, n)
                pos[i:end] = d
                i = end
            else:
                i += 1

        # Volatility-targeted size (always positive).
        with np.errstate(divide="ignore", invalid="ignore"):
            size = params.vol_target / atr_pct
        size = np.where(np.isfinite(size), size, 1.0)
        size = np.clip(size, params.size_floor, params.size_cap)

        df = pd.DataFrame(index=idx)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
