from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AroonRegimeParams:
    window: int = 25
    profit_target: float = 0.05


class GeneratedStrategy(BaseStrategy[AroonRegimeParams]):
    """Regime-switching long-only strategy.

    Regime is classified purely by the temporal recency of high-low range
    extremes: how many bars ago the rolling-window highest high occurred
    versus how many bars ago the rolling-window lowest low occurred. When the
    newest high becomes more recent than the newest low, the regime is 'up'.
    The bar on which that recency ordering flips up is treated as a
    propagating regime front (a traffic-shockwave-style boundary) and is the
    entry. Exit is the first of a fixed profit target or a ~2-week time-stop.
    """

    strategy_id = "gen_a1_1778914014"

    MAX_HOLD_BARS = 10  # ~2 trading weeks; derived, not tunable

    @classmethod
    def params_type(cls):
        return AroonRegimeParams

    def warmup_bars(self, params: AroonRegimeParams) -> int:
        return int(params.window) + 1

    def indicators(self, data: pd.DataFrame, params: AroonRegimeParams) -> pd.DataFrame:
        w = int(params.window)
        if w < 2:
            w = 2
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        n = len(high)

        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)

        if n >= w:
            sw_h = np.lib.stride_tricks.sliding_window_view(high, w)
            sw_l = np.lib.stride_tricks.sliding_window_view(low, w)
            # Reverse each window so index 0 is the most recent bar; argmax/
            # argmin then directly yields bars-since the most recent extreme.
            bars_since_high = sw_h[:, ::-1].argmax(axis=1)
            bars_since_low = sw_l[:, ::-1].argmin(axis=1)
            au = 100.0 * (w - bars_since_high) / w
            ad = 100.0 * (w - bars_since_low) / w
            aroon_up[w - 1:] = au
            aroon_down[w - 1:] = ad

        ind = pd.DataFrame(index=data.index)
        ind["aroon_up"] = aroon_up
        ind["aroon_down"] = aroon_down

        regime = (ind["aroon_up"] > ind["aroon_down"]).astype(float)
        regime[ind["aroon_up"].isna()] = np.nan
        ind["regime_up"] = regime
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AroonRegimeParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        regime = indicators["regime_up"].to_numpy(dtype=float)
        n = len(close)

        pt = float(params.profit_target)
        max_hold = int(self.MAX_HOLD_BARS)

        raw = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        bars_held = 0
        prev_regime = 0.0

        for i in range(n):
            r = regime[i]
            if np.isnan(r):
                # Warmup: no valid regime yet.
                prev_regime = 0.0
                continue

            if in_pos:
                bars_held += 1
                gain = close[i] / entry_price - 1.0 if entry_price > 0.0 else 0.0
                if gain >= pt or bars_held >= max_hold:
                    raw[i] = 0
                    in_pos = False
                    entry_price = 0.0
                    bars_held = 0
                else:
                    raw[i] = 1
            else:
                # Enter on a fresh regime flip up: the newest high has just
                # overtaken the newest low in recency (the front passes).
                if r == 1.0 and prev_regime == 0.0:
                    raw[i] = 1
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                else:
                    raw[i] = 0

            prev_regime = r

        signal = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
