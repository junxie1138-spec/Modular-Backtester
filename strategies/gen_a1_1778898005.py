from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA1Params:
    roc_window: int = 10
    stop_k: float = 2.5


class GeneratedStrategy(BaseStrategy[GenA1Params]):
    strategy_id = "gen_a1_1778898005"

    _ATR_WINDOW = 14
    _MAX_HOLD = 20

    @classmethod
    def params_type(cls):
        return GenA1Params

    def warmup_bars(self, params: GenA1Params) -> int:
        w = max(int(params.roc_window), 2)
        return 2 * w + self._ATR_WINDOW + 2

    def indicators(self, data: pd.DataFrame, params: GenA1Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        w = max(int(params.roc_window), 2)

        # Rate of change and its bar-to-bar acceleration (change in trend speed).
        roc = close.pct_change(w)
        accel = roc.diff()

        # Yield-strength band: dispersion of acceleration itself. Accelerations
        # inside this band are elastic noise; beyond it the trend has yielded.
        yield_band = accel.rolling(w).std()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self._ATR_WINDOW).mean()

        return pd.DataFrame(
            {
                "roc": roc,
                "accel": accel,
                "yield_band": yield_band,
                "atr": atr,
            },
            index=data.index,
        )

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        roc = indicators["roc"].to_numpy(dtype=float)
        accel = indicators["accel"].to_numpy(dtype=float)
        yield_band = indicators["yield_band"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        stop_level = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                # Fixed volatility-stop: ATR captured at entry, never moved.
                if close[i] <= stop_level or bars_held >= self._MAX_HOLD:
                    in_pos = False
                    stop_level = 0.0
                    bars_held = 0
                    raw[i] = 0
                else:
                    raw[i] = 1
            else:
                ready = (
                    np.isfinite(accel[i])
                    and np.isfinite(yield_band[i])
                    and np.isfinite(roc[i])
                    and np.isfinite(atr[i])
                )
                if (
                    ready
                    and atr[i] > 0.0
                    and yield_band[i] > 0.0
                    and roc[i] > 0.0
                    and accel[i] > yield_band[i]
                ):
                    in_pos = True
                    stop_level = close[i] - float(params.stop_k) * atr[i]
                    bars_held = 0
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
