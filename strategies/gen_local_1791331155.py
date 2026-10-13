from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    variance_threshold: float = 1.35
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791331155"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 26

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.variance_threshold) or params.variance_threshold <= 1.0:
            raise ValueError("variance_threshold must be finite and greater than one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["log_return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        out["return5"] = out["log_return"].rolling(5, min_periods=5).sum()
        out["return10"] = out["log_return"].rolling(10, min_periods=10).sum()
        out["daily_vol"] = out["log_return"].rolling(20, min_periods=20).std(ddof=0)
        out["block_vol"] = out["return5"].rolling(20, min_periods=20).std(ddof=0)
        denominator = 5.0 * out["daily_vol"].pow(2)
        out["variance_ratio"] = out["block_vol"].pow(2).div(denominator.where(denominator > 0))
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["enter"] = (
            (out["variance_ratio"] > params.variance_threshold)
            & (out["variance_ratio"].shift(1) <= params.variance_threshold)
            & (out["return5"] > 0)
            & (out["return10"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["enter"].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            if not active and i > 0 and raw[i - 1] == 1:
                if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atrs[i - 1]) or atrs[i - 1] <= 0:
                    continue
                active = True
                armed = False
                entry = peak = opens[i]
                stop = entry - 2.0 * atrs[i - 1]
                age = 0
            if active:
                age += 1
                if not np.isfinite(lows[i]) or not np.isfinite(closes[i]) or lows[i] <= stop or age >= 10:
                    active = False
                    continue
                peak = max(peak, closes[i])
                armed = armed or closes[i] >= entry * 1.015
                if armed:
                    stop = max(stop, entry)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atrs[i])
                raw[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        # Stops are evaluated at the close; every decision executes next bar.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    sample = pd.DataFrame({
        "open": [100, 100, 100, 102, 104, 101],
        "high": [101, 101, 103, 105, 105, 102],
        "low": [99, 99, 101, 103, 100, 99],
        "close": [100, 100, 102, 104, 101, 100],
        "volume": [1000] * 6,
    }, index=index)
    derived = pd.DataFrame({"atr": 1.0, "enter": [True, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy().generate_signals(sample, derived, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
