from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAcceptanceParams:
    range_window: int = 10
    atr_window: int = 14
    range_cap: float = 0.85
    gap_min_atr: float = 0.20
    breakeven_pct: float = 0.01
    trail_atr: float = 1.2
    initial_stop_atr: float = 1.5
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GapAcceptanceParams]):
    strategy_id = "gen_local-w2_1791239824"

    @classmethod
    def params_type(cls) -> type[GapAcceptanceParams]:
        return GapAcceptanceParams

    def warmup_bars(self, params: GapAcceptanceParams) -> int:
        return int(max(200, params.range_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapAcceptanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["day_range"] = data["high"] - data["low"]
        out["high_gap_distance"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap_distance"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["day_range", "high_gap_distance", "low_gap_distance"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["range_mean"] = out["day_range"].shift(1).rolling(params.range_window, min_periods=params.range_window).mean()
        out["range_ratio"] = out["day_range"] / out["range_mean"].where(out["range_mean"] > 0)
        out["gap"] = data["open"] - out["previous_close"]
        out["gap_atr"] = out["gap"] / out["previous_atr"].where(out["previous_atr"] > 0)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        valid = np.isfinite(out).all(axis=1) & (out["atr"] > 0) & (out["day_range"] > 0)
        compact = out["range_ratio"] <= params.range_cap
        long_entry = valid & compact & (out["gap_atr"] >= params.gap_min_atr) & (data["low"] > out["previous_high"]) & (data["close"] > data["open"]) & (data["close"] > out["ma200"])
        short_entry = valid & compact & (out["gap_atr"] <= -params.gap_min_atr) & (data["high"] < out["previous_low"]) & (data["close"] < data["open"]) & (data["close"] < out["ma200"])
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapAcceptanceParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        side = 0
        pending = False
        entry_price = 0.0
        entry_atr = 0.0
        stop = 0.0
        activated = False
        age = 0

        for i in range(len(data)):
            if pending:
                # The preceding close's decision fills at this session's open.
                entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    side = 0
                else:
                    # Signed price makes both long and short stop ratchets increase.
                    stop = side * entry_price - params.initial_stop_atr * entry_atr
                pending = False

            if side:
                age += 1
                close = closes[i]
                atr = atrs[i]
                if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
                    side = 0
                else:
                    signed_close = side * close
                    stopped = signed_close <= stop
                    if not activated and side * (close / entry_price - 1.0) >= params.breakeven_pct:
                        activated = True
                    if activated:
                        stop = max(stop, side * entry_price, signed_close - params.trail_atr * atr)
                    # Stops are evaluated at closes; exits fill at the next open.
                    if stopped or signed_close <= stop or age >= params.hold_days:
                        side = 0
                decisions[i] = side
                continue

            if entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                side = int(entries[i])
                entry_atr = atrs[i]
                pending = True
                activated = False
                age = 0
                decisions[i] = side

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapAcceptanceParams()
    index = pd.date_range("2020-01-01", periods=4)
    for side in (1, -1):
        data = pd.DataFrame({"open": [100.0] * 4, "close": [100.0, 100.0 + side * 2.0, 100.0, 100.0]}, index=index)
        indicators = pd.DataFrame({"atr": [1.0] * 4, "entry": [side, 0, 0, 0]}, index=index)
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
        assert result.data["signal"].tolist() == [0, side, side, 0]
        assert (result.data["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
