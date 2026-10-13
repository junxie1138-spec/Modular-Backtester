from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownRepairParams:
    depth_atr: float = 1.5
    trail_atr: float = 1.0


class GeneratedStrategy(BaseStrategy[DrawdownRepairParams]):
    strategy_id = "gen_local-w2_1790940777"

    @classmethod
    def params_type(cls):
        return DrawdownRepairParams

    def warmup_bars(self, params: DrawdownRepairParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: DrawdownRepairParams) -> pd.DataFrame:
        if not np.isfinite(params.depth_atr) or params.depth_atr <= 0:
            raise ValueError("depth_atr must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        close = data["close"]
        previous_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        tr = tr.where(previous_close.notna())
        out = pd.DataFrame(index=data.index)
        out["atr"] = tr.rolling(14, min_periods=14).mean().replace(0.0, np.nan)
        out["peak"] = close.rolling(20, min_periods=20).max()
        out["depth"] = out["peak"] - close
        out["previous_depth_atr"] = out["depth"].shift(1) / out["atr"].shift(1)
        out["repair_atr"] = (out["depth"].shift(1) - out["depth"]) / out["atr"].shift(1)
        out["peak_change"] = out["peak"].diff()
        out["price_change"] = close.diff()
        ready = np.isfinite(out).all(axis=1)
        eligible = ready & (out["previous_depth_atr"] >= params.depth_atr) & (out["repair_atr"] >= 0.25)
        genuine = eligible & (out["peak_change"] == 0.0) & (out["price_change"] > 0.0)
        expiration = eligible & (out["peak_change"] < 0.0) & (out["price_change"] < 0.0)
        out["entry"] = np.select([genuine, expiration], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownRepairParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        highest = lowest = np.nan
        stop = -np.inf
        short_stop = np.inf
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                # The previous close's decision fills at this bar's open.
                if age == 0:
                    if not np.isfinite(opens[i]):
                        position = 0
                        continue
                    highest = max(opens[i], close)
                    lowest = min(opens[i], close)
                    stop = -np.inf
                    short_stop = np.inf
                highest = max(highest, close)
                lowest = min(lowest, close)
                stop = max(stop, highest - params.trail_atr * atr)
                # Also protect shorts with a mirrored, downward-ratcheting stop.
                short_stop = min(short_stop, lowest + params.trail_atr * atr)
                age += 1
                trailing_exit = close < stop
                short_exit = position == -1 and close > short_stop
                if trailing_exit or short_exit or age >= 2:
                    position = 0
                    age = 0
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = DrawdownRepairParams(trail_atr=0.5)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": [100, 101, 99, 100, 100, 102], "close": [100, 99, 99, 100, 102, 102]}, index=index)
    derived = pd.DataFrame({"atr": 1.0, "entry": [1, 0, 0, -1, 0, 0]}, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    flat = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    derived["entry"] = [1, 0, 0, 0, 0, 0]
    result = strategy.generate_signals(flat, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0]
