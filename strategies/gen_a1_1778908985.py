from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class HealedDrawdownParams:
    peak_window: int = 60
    trend_ma: int = 100
    atr_window: int = 14
    atr_k: float = 2.5
    min_dd_depth: float = 0.05
    refractory_bars: int = 5
    max_hold_bars: int = 10
    base_size: float = 0.4
    size_scale: float = 6.0
    min_size: float = 0.3
    max_size: float = 1.0


class GeneratedStrategy(BaseStrategy[HealedDrawdownParams]):
    """Long-only momentum: enter when a deep drawdown episode fully heals into a
    new rolling high; position size scales with the depth of the healed
    drawdown; exit on a fixed ATR volatility stop set at entry."""

    strategy_id = "gen_a1_1778908985"

    @classmethod
    def params_type(cls) -> type[HealedDrawdownParams]:
        return HealedDrawdownParams

    @staticmethod
    def warmup_bars(params: HealedDrawdownParams) -> int:
        return int(max(params.peak_window, params.trend_ma, params.atr_window + 1)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: HealedDrawdownParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        roll_max = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        drawdown = close / roll_max - 1.0

        true_range = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()

        trend_ma = close.rolling(params.trend_ma, min_periods=params.trend_ma).mean()

        out = pd.DataFrame(index=data.index)
        out["drawdown"] = drawdown
        out["atr"] = atr
        out["trend_ma"] = trend_ma
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: HealedDrawdownParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        dd = indicators["drawdown"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        tma = indicators["trend_ma"].to_numpy(dtype=float)
        n = len(close)

        sig = np.zeros(n, dtype=int)
        sz = np.ones(n, dtype=float)

        tol = 1e-9
        in_pos = False
        stop = 0.0
        bars_held = 0
        refr = 0
        pos_size = float(params.base_size)
        episode_min_dd = 0.0  # most negative drawdown seen since last new high

        lo = max(float(params.min_size), 1e-6)
        hi = max(float(params.max_size), lo)

        for i in range(n):
            if np.isnan(dd[i]) or np.isnan(atr[i]) or np.isnan(tma[i]):
                continue
            if refr > 0:
                refr -= 1

            # Track the depth of the current underwater episode. On the bar that
            # prints a fresh rolling high, the episode is complete and its
            # healed depth becomes available for one bar only.
            recovered = 0.0
            if dd[i] >= -tol:
                recovered = -episode_min_dd
                episode_min_dd = 0.0
            elif dd[i] < episode_min_dd:
                episode_min_dd = dd[i]

            if in_pos:
                bars_held += 1
                if close[i] <= stop or bars_held >= params.max_hold_bars:
                    in_pos = False
                    sig[i] = 0
                    refr = int(params.refractory_bars)
                else:
                    sig[i] = 1
                    sz[i] = pos_size
            else:
                if (
                    refr == 0
                    and recovered >= params.min_dd_depth
                    and close[i] > tma[i]
                    and atr[i] > 0.0
                ):
                    in_pos = True
                    stop = close[i] - params.atr_k * atr[i]
                    bars_held = 0
                    raw = params.base_size + params.size_scale * recovered
                    pos_size = min(max(raw, lo), hi)
                    sig[i] = 1
                    sz[i] = pos_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(sig, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = pd.Series(sz, index=data.index).shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
