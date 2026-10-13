from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    outbreak_ratio: float = 1.35
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791288387"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 29

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.outbreak_ratio) or params.outbreak_ratio <= 1.0:
            raise ValueError("outbreak_ratio must exceed 1")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["range_reference"] = out["range"].rolling(20, min_periods=20).mean().shift(1)
        out["range_ratio"] = out["range"] / out["range_reference"].where(out["range_reference"] > 0.0)
        out["wide"] = out["range_ratio"].gt(params.outbreak_ratio).astype(float).where(out["range_ratio"].notna())
        out["cluster"] = out["wide"].rolling(4, min_periods=4).sum().shift(1)
        out["earlier_cluster"] = out["cluster"].shift(4)
        out["close_location"] = (data["close"] - data["low"]) / out["range"].where(out["range"] > 0.0)
        out["range_decay"] = out["range"] / out["range"].shift(1).where(out["range"].shift(1) > 0.0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            out["cluster"].ge(3.0)
            & out["earlier_cluster"].le(1.0)
            & out["range_decay"].lt(0.85)
            & out["close_location"].ge(0.65)
            & data["low"].ge(data["low"].shift(1))
            & data["close"].gt(previous_close)
            & out["atr"].gt(0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        count = len(data)
        desired = np.zeros(count, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        entry_bar = 0

        for i in range(count):
            valid = (
                np.isfinite(opens[i]) and opens[i] > 0.0
                and np.isfinite(highs[i]) and np.isfinite(lows[i])
                and np.isfinite(closes[i]) and closes[i] > 0.0
                and lows[i] <= min(opens[i], closes[i])
                and highs[i] >= max(opens[i], closes[i])
                and np.isfinite(atr[i]) and atr[i] > 0.0
            )
            if not valid:
                active = False
                continue

            # Yesterday's close decision enters at today's open.
            if not active and i > 0 and desired[i - 1] == 1:
                if not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0.0:
                    continue
                active = True
                armed = False
                entry_price = opens[i]
                entry_bar = i
                stop = entry_price - 2.0 * atr[i - 1]

            if active:
                # Existing stops are observed during this bar; fills remain next-open.
                exit_now = opens[i] <= stop or lows[i] <= stop
                if not exit_now:
                    if highs[i] >= entry_price * 1.008:
                        armed = True
                    if armed:
                        previous_stop = stop
                        stop = max(stop, entry_price, highs[i] - params.trail_atr * atr[i])
                        assert stop >= previous_stop
                        # New ratchets cannot assume an intrabar high-before-low order.
                        exit_now = closes[i] <= stop
                if exit_now or i - entry_bar + 1 >= 5:
                    active = False
                else:
                    desired[i] = 1
                # A stop or time exit cannot immediately re-enter at the same close.
                continue

            if entries[i]:
                desired[i] = 1

        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 100.5, 101.0, 102.0, 101.0],
        "high": [100.5, 100.5, 101.2, 103.0, 102.5, 102.0],
        "low": [99.5, 99.5, 100.2, 100.2, 101.0, 100.0],
        "close": [100.0, 100.0, 100.8, 102.0, 101.2, 101.0],
        "volume": [1000.0] * 6,
    }, index=index)
    features = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, features, None, Params())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    # Breakeven protection must close a trade even with a distant ATR trail.
    data.loc[index[3], ["open", "high", "low", "close"]] = [100.8, 101.0, 99.9, 100.2]
    result = GeneratedStrategy.generate_signals(None, data, features, None, Params(trail_atr=10.0))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
