from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    baseline_window: int = 40
    capacity_window: int = 6
    atr_window: int = 14
    regime_window: int = 200
    compression_ratio: float = 0.70
    capacity_threshold: float = 2.0
    release_multiple: float = 1.15
    breakeven_pct: float = 0.01
    initial_stop_atr: float = 2.0
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = "gen_local-w2_1791312014"

    @classmethod
    def params_type(cls):
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return int(max(params.regime_window, params.atr_window + 1,
                       params.baseline_window + params.capacity_window + 2))

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        p = params
        if min(p.baseline_window, p.capacity_window, p.atr_window, p.max_hold) < 1:
            raise ValueError("Lookbacks and holding limit must be positive.")
        if p.regime_window != 200:
            raise ValueError("The regime filter must use the 200-day moving average.")
        if not all(np.isfinite(v) and v > 0 for v in
                   (p.compression_ratio, p.capacity_threshold, p.release_multiple,
                    p.breakeven_pct, p.initial_stop_atr, p.trail_atr)):
            raise ValueError("Thresholds and stop distances must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["baseline"] = out["tr"].rolling(p.baseline_window).mean().shift(1)
        baseline = out["baseline"].where(out["baseline"] > 0)
        out["atr"] = out["tr"].rolling(p.atr_window).mean()
        out["ma200"] = data["close"].rolling(p.regime_window).mean()
        out["deficit"] = (baseline - out["tr"]).clip(lower=0)
        out["capacity"] = out["deficit"].rolling(p.capacity_window).sum() / baseline
        out["compression"] = out["tr"].rolling(p.capacity_window).mean() / baseline
        out["prior_capacity"] = out["capacity"].shift(1)
        out["prior_compression"] = out["compression"].shift(1)
        out["upper"] = data["high"].rolling(p.capacity_window).max().shift(1)
        out["lower"] = data["low"].rolling(p.capacity_window).min().shift(1)
        out["ready"] = np.isfinite(out).all(axis=1) & (out["atr"] > 0)
        release = (
            out["ready"]
            & (out["prior_capacity"] >= p.capacity_threshold)
            & (out["prior_compression"] <= p.compression_ratio)
            & (out["tr"] >= p.release_multiple * baseline)
        )
        out["entry"] = np.select([
            release & (data["close"] > out["upper"]) & (data["close"] > out["ma200"]),
            release & (data["close"] < out["lower"]) & (data["close"] < out["ma200"])
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CapacityParams) -> SignalFrame:
        # Stops are close-time decisions, executed at the following bar's open.
        # Signed prices make the protective stop increase for either direction.
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry = np.nan
        stop = -np.inf
        armed = False
        age = 0
        for i, (opening, high, low, close) in enumerate(prices):
            if not ready[i] or not np.isfinite(prices[i]).all() or min(prices[i]) <= 0:
                side = 0
                entry = np.nan
                continue
            if side:
                if np.isnan(entry):
                    # Yesterday's decision enters at today's observable opening.
                    entry = opening
                    stop = side * entry - params.initial_stop_atr * atrs[i - 1]
                    armed = False
                    age = 0
                age += 1
                adverse = low if side == 1 else -high
                if adverse <= stop or age >= params.max_hold:
                    side = 0
                    entry = np.nan
                    continue
                favorable_close = side * close
                if side * (close - entry) / entry >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, side * entry,
                               favorable_close - params.trail_atr * atrs[i])
                # A newly ratcheted close-based stop applies from the next bar.
            elif entries[i]:
                side = int(entries[i])
                entry = np.nan
            signals[i] = side
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = CapacityParams()
    for side in (1, -1):
        closes = [100.0, 100.0 + 2.0 * side, 100.0, 100.0]
        data = pd.DataFrame({"open": [100.0] * 4, "high": [100.5, 103.0 if side == 1 else 101.0, 101.0, 101.0],
                             "low": [99.5, 99.0 if side == 1 else 97.0, 99.0, 99.0],
                             "close": closes, "volume": [1000.0] * 4},
                            index=pd.date_range("2020-01-01", periods=4))
        ind = pd.DataFrame({"atr": 1.0, "ready": True, "entry": [side, 0, 0, 0]}, index=data.index)
        result = GeneratedStrategy.generate_signals(None, data, ind, None, p).data
        assert result["signal"].tolist() == [0, side, side, 0]
        assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
