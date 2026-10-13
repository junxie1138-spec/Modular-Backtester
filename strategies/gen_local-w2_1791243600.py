from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    compression_ratio: float = 0.75
    innovation_threshold: float = 0.04


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791243600"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 26

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.compression_ratio < 1.0):
            raise ValueError("compression_ratio must be between zero and one")
        if not (np.isfinite(params.innovation_threshold) and params.innovation_threshold > 0.0):
            raise ValueError("innovation_threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["recent_range"] = out["true_range"].rolling(5, min_periods=5).mean().shift(1)
        out["background_range"] = out["true_range"].rolling(20, min_periods=20).mean().shift(5)
        out["compression"] = out["recent_range"] / out["background_range"].replace(0.0, np.nan)
        # Seven-point least-squares slope, oldest weight -3, newest +3.
        out["slope"] = sum((3 - lag) * out["midpoint"].shift(lag) for lag in range(7)) / 28.0
        out["innovation"] = out["slope"].diff() / out["atr"].replace(0.0, np.nan)
        out["previous_innovation"] = out["innovation"].shift(1)
        out["ceiling"] = data["high"].rolling(10, min_periods=10).max().shift(1)
        out["floor"] = data["low"].rolling(10, min_periods=10).min().shift(1)
        out = out.replace([np.inf, -np.inf], np.nan)
        valid_prices = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        out["ready"] = out.notna().all(axis=1) & valid_prices & (out["atr"] > 0.0) & (data["low"] > 0.0)
        eligible = (
            out["ready"]
            & (out["compression"] < params.compression_ratio)
            & (data["close"] > out["floor"])
            & (data["close"] < out["ceiling"])
        )
        threshold = params.innovation_threshold
        long_entry = eligible & (out["slope"] > 0.0) & (out["innovation"] > threshold) & (out["previous_innovation"] <= threshold)
        short_entry = eligible & (out["slope"] < 0.0) & (out["innovation"] < -threshold) & (out["previous_innovation"] >= -threshold)
        out["candidate"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        ready = indicators["ready"].to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        direction = 0
        pending = False
        entry = 0.0
        stop = 0.0
        age = 0
        activated = False
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if direction:
                if not (np.isfinite(prices[i]).all() and opening > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0):
                    direction = 0
                    pending = False
                    continue
                if pending:
                    # The preceding close's decision fills at this bar's open.
                    entry = opening
                    stop = direction * entry - 1.5 * atr[i - 1]
                    age = 0
                    activated = False
                    pending = False
                age += 1
                adverse = low if direction == 1 else -high
                oriented_close = direction * close
                # Test only the stop established before this bar; its OHLC order is unknown.
                if adverse <= stop or age >= 4:
                    direction = 0
                    continue
                if direction * (close - entry) >= 0.008 * entry:
                    activated = True
                if activated:
                    # Signed-price stops only increase; short price stops only decrease.
                    stop = max(stop, direction * entry, oriented_close - 1.5 * atr[i])
                raw[i] = direction
            elif ready[i] and candidates[i] != 0:
                direction = int(candidates[i])
                pending = True
                raw[i] = direction
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        # Exit decisions, including stop breaches, execute at the following open.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    for direction in (1, -1):
        close = np.array([100.0, 100.0, 100.0 + 2.0 * direction, 100.0, 100.0, 100.0])
        data = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"atr": 1.0, "ready": True, "candidate": [direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, None, Params()).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0]
        data[["open", "close"]] = 100.0
        data["high"] = 100.2
        data["low"] = 99.8
        result = strategy.generate_signals(data, indicators, None, Params()).data
        assert result["signal"].tolist() == [0, direction, direction, direction, direction, 0]
    derived = strategy.indicators(data, Params())
    assert derived["candidate"].eq(0).all()
    assert strategy.warmup_bars(Params()) >= 26


if __name__ == "__main__":
    _self_check()
