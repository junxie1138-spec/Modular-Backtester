from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    shock_threshold: float = 1.25
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791331495"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.shock_threshold <= 0 or params.trail_atr <= 0:
            raise ValueError("Parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["mean"] = out["midpoint"].rolling(12, min_periods=12).mean()
        weighted = sum((11 - j) * out["midpoint"].shift(j) for j in range(12))
        out["slope"] = (weighted - 66.0 * out["mean"]) / 143.0
        out["endpoint"] = out["mean"] + 5.5 * out["slope"]
        out["slope_innovation"] = (
            out["midpoint"] - out["endpoint"].shift(1) - out["slope"].shift(1)
        ) / out["atr"].replace(0.0, np.nan)
        out["previous_innovation"] = out["slope_innovation"].shift(1)
        out["close_location"] = (
            (data["close"] - data["low"])
            / (data["high"] - data["low"]).replace(0.0, np.nan)
        )
        stable = out["slope"].shift(1).abs() < 0.3 * out["atr"]
        recovering = (
            (out["previous_innovation"] < -params.shock_threshold)
            & (out["slope_innovation"] > out["previous_innovation"])
            & (out["slope_innovation"] < 0)
            & (out["close_location"] > 0.5)
        )
        fading = (
            (out["previous_innovation"] > params.shock_threshold)
            & (out["slope_innovation"] < out["previous_innovation"])
            & (out["slope_innovation"] > 0)
            & (out["close_location"] < 0.5)
        )
        out["entry"] = np.select([stable & recovering, stable & fading], [1, -1], default=0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        side = pending = age = 0
        entry_price = stop = 0.0
        armed = False
        for i in range(n):
            if pending:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    side = pending
                    entry_price = opens[i]
                    stop = side * entry_price - 1.5 * atr[i - 1]
                    age = 0
                    armed = False
                pending = 0
            if side:
                age += 1
                signed_close = side * closes[i]
                valid = np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0
                # Stops are evaluated at the close; exits execute next bar.
                if not valid or signed_close <= stop or age >= 5:
                    side = 0
                else:
                    if signed_close - side * entry_price >= 0.0075 * entry_price:
                        armed = True
                    if armed:
                        # Signed stops only rise; short-price stops only fall.
                        stop = max(stop, side * entry_price, signed_close - params.trail_atr * atr[i])
                decisions[i] = side
            elif entries[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0:
                pending = int(entries[i])
                decisions[i] = pending
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        prices = 100.0 + direction * np.array([0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        data = pd.DataFrame(
            {"open": 100.0, "high": prices + 1.0, "low": prices - 1.0,
             "close": prices, "volume": 1000.0},
            index=pd.date_range("2020-01-01", periods=len(prices)),
        )
        indicators = pd.DataFrame(
            {"atr": 1.0, "entry": [direction, 0, 0, 0, 0, 0, 0]}, index=data.index
        )
        result = strategy.generate_signals(data, indicators, None, Params()).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0, 0]
        assert (result["size"] > 0).all()
        data["close"] = 100.0
        result = strategy.generate_signals(data, indicators, None, Params()).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, direction, 0]


if __name__ == "__main__":
    _self_check()
