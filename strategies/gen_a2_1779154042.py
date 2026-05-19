from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy

# Fixed (non-tunable) mechanics to honor the <=2 tunable-param twist.
K_ATR = 2.5      # fixed volatility-stop multiplier
HOLD_BARS = 19   # ~3-4 week max holding horizon (trading days)


@dataclass(slots=True)
class GeneratedParams:
    window: int = 25
    entry_z: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a2_1779154042"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(params.window) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        window = max(int(params.window), 2)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ma = close.rolling(window).mean()
        sd = close.rolling(window).std(ddof=0)
        sd = sd.where(sd > 0.0, np.nan)
        z = (close - ma) / sd
        dz = z.diff()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window).mean()

        out = pd.DataFrame(index=data.index)
        out["z"] = z
        out["dz"] = dz
        out["atr"] = atr
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        entry_z = float(params.entry_z)

        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        dz = indicators["dz"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw = np.zeros(n, dtype=int)

        pos = 0
        entry_price = 0.0
        stop = 0.0
        bars_held = 0

        for i in range(n):
            if pos == 0:
                ready = (
                    not np.isnan(z[i])
                    and not np.isnan(dz[i])
                    and not np.isnan(atr[i])
                    and atr[i] > 0.0
                )
                if ready:
                    # Long: deeply displaced below the mean, epidemic past its
                    # peak (z first difference has turned back up).
                    if z[i] <= -entry_z and dz[i] > 0.0:
                        pos = 1
                        entry_price = close[i]
                        stop = entry_price - K_ATR * atr[i]
                        bars_held = 0
                    # Short: deeply displaced above the mean, displacement
                    # epidemic past its peak (z first difference turned down).
                    elif z[i] >= entry_z and dz[i] < 0.0:
                        pos = -1
                        entry_price = close[i]
                        stop = entry_price + K_ATR * atr[i]
                        bars_held = 0
            else:
                bars_held += 1
                if pos == 1:
                    hit_stop = close[i] <= stop
                    reverted = (not np.isnan(z[i])) and z[i] >= 0.0
                    if hit_stop or reverted or bars_held >= HOLD_BARS:
                        pos = 0
                else:
                    hit_stop = close[i] >= stop
                    reverted = (not np.isnan(z[i])) and z[i] <= 0.0
                    if hit_stop or reverted or bars_held >= HOLD_BARS:
                        pos = 0

            raw[i] = pos

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
