from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    depth: float = 0.04
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791337448"

    @classmethod
    def params_type(cls) -> type[DrawdownParams]:
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return 73

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        if not np.isfinite(params.depth) or not 0.0 < params.depth < 1.0:
            raise ValueError("depth must be finite and between zero and one")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["peak"] = data["close"].rolling(63, min_periods=63).max()
        out["drawdown"] = 1.0 - data["close"] / out["peak"].replace(0.0, np.nan)
        out["previous_drawdown"] = out["drawdown"].shift(1)
        out["recent_depth"] = out["previous_drawdown"].rolling(10, min_periods=10).max()
        out["momentum"] = data["close"] / data["close"].shift(20).replace(0.0, np.nan) - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        ready = out[["drawdown", "previous_drawdown", "recent_depth", "momentum", "atr"]].notna().all(axis=1)
        long_entry = ready & (out["previous_drawdown"] >= params.depth / 2.0) & (out["drawdown"] < params.depth / 2.0) & (out["recent_depth"] >= params.depth) & (out["momentum"] > 0.0)
        short_entry = ready & (out["previous_drawdown"] < params.depth) & (out["drawdown"] >= params.depth) & (out["momentum"] < 0.0)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownParams) -> SignalFrame:
        count = len(data)
        targets = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        position = 0
        pending = 0
        entry_price = 0.0
        entry_bar = 0
        best_gain = 0.0
        stop_gain = 0.0
        armed = False

        for i in range(count):
            if position != 0 and pending == 0:
                position = 0
            if position == 0 and pending != 0:
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0.0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0.0:
                    position = pending
                    entry_price = opens[i]
                    entry_bar = i
                    best_gain = 0.0
                    stop_gain = -params.trail_atr * atrs[i - 1]
                    armed = False
                else:
                    pending = 0

            if position != 0:
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0.0:
                    pending = 0
                else:
                    gain = position * (closes[i] - entry_price)
                    best_gain = max(best_gain, gain)
                    armed = armed or best_gain >= 0.01 * entry_price
                    if armed:
                        # The stop ratchets upward in signed profit coordinates for both directions.
                        stop_gain = max(stop_gain, 0.0, best_gain - params.trail_atr * atrs[i])
                    pending = 0 if gain <= stop_gain or i - entry_bar + 1 >= 18 else position
            else:
                pending = int(entries[i]) if np.isfinite(closes[i]) and closes[i] > 0.0 and np.isfinite(atrs[i]) and atrs[i] > 0.0 else 0
            targets[i] = pending

        df = pd.DataFrame({"signal": targets, "size": np.full(count, 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    t = np.arange(240, dtype=float)
    close = 100.0 * np.exp(0.08 * np.sin(2.0 * np.pi * t / 40.0))
    sample = pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": np.full(240, 1000000.0)}, index=pd.date_range("2020-01-01", periods=240))
    strategy = GeneratedStrategy()
    params = DrawdownParams()
    derived = strategy.indicators(sample, params)
    result = strategy.generate_signals(sample, derived, None, params).data
    first = int(np.flatnonzero(derived["entry"].to_numpy())[0])
    assert result["signal"].iloc[first] == 0
    assert result["signal"].iloc[first + 1] == derived["entry"].iloc[first]
    assert set(result["signal"].unique()) <= {-1, 0, 1}
    assert (result["size"] > 0.0).all()
    prefix = sample.iloc[:160]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    pd.testing.assert_frame_equal(result.iloc[:160], prefix_result)
