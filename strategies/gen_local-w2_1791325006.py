from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    expansion_ratio: float = 1.8
    stop_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791325006"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.expansion_ratio) or params.expansion_ratio <= 1.0:
            raise ValueError("expansion_ratio must be finite and greater than one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["range"] = data["high"] - data["low"]
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["prior_range"] = out["range"].shift(1)
        out["close_location"] = (
            (data["close"] - data["low"]) / out["range"].where(out["range"] > 0.0)
        )
        out["contracting"] = (
            (out["range"].shift(1) < out["range"].shift(2))
            & (out["range"].shift(2) < out["range"].shift(3))
        )
        out["balanced_dilation"] = (
            out["contracting"]
            & (out["prior_range"] > 0.0)
            & (out["range"] >= params.expansion_ratio * out["prior_range"])
            & (data["high"] > data["high"].shift(1))
            & (data["low"] < data["low"].shift(1))
            & ((out["midpoint"] - out["midpoint"].shift(1)).abs()
               <= 0.25 * out["prior_range"])
            & out["atr"].gt(0.0)
        )
        out["candidate"] = np.select([
            out["balanced_dilation"] & out["close_location"].le(0.2),
            out["balanced_dilation"] & out["close_location"].ge(0.8),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        position = 0
        entry = np.nan
        stop_distance = 0.0
        age = 0
        for i in range(len(data)):
            if position:
                # The preceding close requested entry; this bar supplies its open.
                if age == 0:
                    entry = opens[i]
                age += 1
                valid = np.isfinite(entry) and entry > 0.0 and np.isfinite(closes[i])
                stopped = valid and (
                    (position == 1 and closes[i] < entry - stop_distance)
                    or (position == -1 and closes[i] > entry + stop_distance)
                )
                if not valid or stopped or age >= 2:
                    position = 0
                    age = 0
                    entry = np.nan
                # Exit decisions remain flat even if a fresh candidate exists.
            elif (
                candidates[i] != 0
                and np.isfinite(closes[i])
                and closes[i] > 0.0
                and np.isfinite(atrs[i])
                and atrs[i] > 0.0
            ):
                position = int(candidates[i])
                stop_distance = params.stop_atr * atrs[i]
                age = 0
                entry = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 96.0,
        "close": [100.0, 98.0, 100.0, 100.0, 102.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0, "candidate": [1, 0, 0, -1, 0, 0],
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0.0).all()
