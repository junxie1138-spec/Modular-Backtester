from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    min_depth: float = 1.5
    stop_atr: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in (self.min_depth, self.stop_atr)):
            raise ValueError("Parameters must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791204045"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 25

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out["peak"] = close.rolling(20, min_periods=20).max()
        out["daily_range"] = data["high"] - data["low"]
        out["true_range"] = pd.concat(
            [out["daily_range"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["depth"] = (out["peak"] - close) / out["atr"].where(out["atr"] > 0)
        out["depth_baseline"] = out["depth"].rolling(5, min_periods=5).mean().shift(1)
        out["squeeze_span"] = (
            data["high"].rolling(3, min_periods=3).max()
            - data["low"].rolling(3, min_periods=3).min()
        )
        out["range_baseline"] = out["daily_range"].rolling(20, min_periods=20).mean().shift(3)
        out["compressed"] = (
            (out["range_baseline"] > 0)
            & (out["squeeze_span"] <= 1.25 * out["range_baseline"])
        ).fillna(False)
        eligible = out["compressed"] & (out["depth"] >= params.min_depth)
        out["candidate"] = np.select(
            [eligible & (out["depth"] < out["depth_baseline"]),
             eligible & (out["depth"] > out["depth_baseline"])],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = age = 0
        high_water, low_water = -np.inf, np.inf
        stop, short_stop = -np.inf, np.inf
        for i, (close, atr) in enumerate(zip(closes, atrs)):
            if not (np.isfinite(close) and np.isfinite(atr) and atr > 0):
                position = 0
                continue
            if position:
                age += 1
                if age == 1:
                    # The preceding decision fills at this bar's open.
                    if not (np.isfinite(opens[i]) and opens[i] > 0):
                        position = 0
                        continue
                    stop = opens[i] - params.stop_atr * atrs[i - 1]
                    short_stop = opens[i] + params.stop_atr * atrs[i - 1]
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                stop = max(stop, high_water - params.stop_atr * atr)
                short_stop = min(short_stop, low_water + params.stop_atr * atr)
                # The required high-water stop applies to both directions.
                # Shorts also have a mirrored low-water protective stop.
                if close <= stop or (position < 0 and close >= short_stop) or age >= 2:
                    position = 0
            elif candidates[i]:
                position = int(candidates[i])
                age = 0
                high_water, low_water = -np.inf, np.inf
                stop, short_stop = -np.inf, np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=4)
    for direction, second_close, expected in (
        (1, 98.0, [0, 1, 0, 0]), (-1, 102.0, [0, -1, 0, 0]),
        (1, 100.0, [0, 1, 1, 0]), (-1, 100.0, [0, -1, -1, 0])
    ):
        data = pd.DataFrame({"open": 100.0, "close": [100.0, second_close, 100.0, 100.0]}, index=index)
        derived = pd.DataFrame({"atr": 1.0, "candidate": [direction, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, derived, None, Params()).data
        assert result["signal"].tolist() == expected
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
