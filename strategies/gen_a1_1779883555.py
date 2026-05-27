from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SpringTensionParams:
    ma_length: int = 20
    stop_k: float = 2.0


class GeneratedStrategy(BaseStrategy[SpringTensionParams]):
    strategy_id = "gen_a1_1779883555"

    @classmethod
    def params_type(cls) -> type[SpringTensionParams]:
        return SpringTensionParams

    @classmethod
    def warmup_bars(cls, params: SpringTensionParams) -> int:
        return int(max(params.ma_length, 14)) + 5

    def indicators(self, data: pd.DataFrame, params: SpringTensionParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        prev_close = close.shift(1)

        n = int(params.ma_length)
        ma = close.rolling(n, min_periods=n).mean()
        std = close.rolling(n, min_periods=n).std(ddof=0)
        std_safe = std.where(std > 0.0, np.nan)
        z = (close - ma) / std_safe

        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()

        return pd.DataFrame({"z": z, "atr": atr}, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SpringTensionParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        raw_signal = np.zeros(n, dtype=np.int64)
        raw_size = np.ones(n, dtype=np.float64)

        entry_threshold = -2.0
        max_hold = 5
        size_cap = 1.5

        in_pos = False
        bars_held = 0
        stop_price = 0.0
        pos_size = 1.0

        stop_k = float(params.stop_k)

        for i in range(n):
            zi = z[i]
            ai = atr[i]
            ci = close[i]

            if in_pos:
                bars_held += 1
                if not np.isfinite(ci):
                    in_pos = False
                    bars_held = 0
                    raw_signal[i] = 0
                    continue
                if ci <= stop_price or bars_held >= max_hold:
                    in_pos = False
                    bars_held = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
                    raw_size[i] = pos_size
                continue

            if not (np.isfinite(zi) and np.isfinite(ai) and np.isfinite(ci)) or ai <= 0.0:
                continue

            if zi <= entry_threshold:
                stretch = abs(zi) / abs(entry_threshold)
                if stretch > size_cap:
                    stretch = size_cap
                pos_size = float(stretch)
                stop_price = ci - stop_k * ai
                bars_held = 0
                in_pos = True
                raw_signal[i] = 1
                raw_size[i] = pos_size

        df = pd.DataFrame({"signal": raw_signal, "size": raw_size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.1)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
