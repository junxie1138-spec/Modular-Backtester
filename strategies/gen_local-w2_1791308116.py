from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 6
    noise_window: int = 20
    baseline_window: int = 12
    atr_window: int = 14
    innovation_z: float = 0.20
    floor_atr: float = 0.35
    breakeven_pct: float = 0.008
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791308116"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.slope_window + 2, params.noise_window + 3,
                       params.baseline_window + 2, params.atr_window + 3))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        p = params
        if min(p.slope_window, p.noise_window, p.baseline_window, p.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not all(np.isfinite(x) and x > 0 for x in
                   (p.innovation_z, p.floor_atr, p.breakeven_pct, p.trail_atr)):
            raise ValueError("Thresholds and stop distances must be finite and positive.")
        if p.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        out["noise"] = out["return"].rolling(p.noise_window).std(ddof=0).shift(1)
        n = p.slope_window
        center = (n - 1) / 2.0
        denominator = n * (n * n - 1) / 12.0
        out["slope"] = sum((j - center) * out["log_close"].shift(n - 1 - j)
                           for j in range(n)) / denominator
        out["predicted_slope"] = out["slope"].shift(1)
        out["slope_noise"] = (out["noise"] / np.sqrt(n)).where(out["noise"] > 0)
        out["innovation"] = (out["slope"] - out["predicted_slope"]) / out["slope_noise"]
        out["previous_innovation"] = out["innovation"].shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(p.atr_window).mean()
        out["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3.0
        out["old_equilibrium"] = out["typical_price"].rolling(p.baseline_window).mean().shift(2)
        out["old_atr"] = out["atr"].shift(2)
        out["support_floor"] = data["low"].rolling(2).min()
        out["support_barrier"] = out["old_equilibrium"] + p.floor_atr * out["old_atr"]
        out["slope_change"] = (
            (out["innovation"] > p.innovation_z)
            & (out["previous_innovation"] <= p.innovation_z)
            & (out["slope"] > 0)
        )
        out["plastic_shift"] = out["support_floor"] > out["support_barrier"]
        out["ready"] = np.isfinite(out[[
            "innovation", "previous_innovation", "slope", "atr",
            "support_floor", "support_barrier"
        ]]).all(axis=1) & (out["atr"] > 0)
        out["entry_gate"] = (
            out["ready"] & out["slope_change"] & out["plastic_shift"]
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        gate = indicators["entry_gate"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    armed = False
                    entry = opens[i]
                    entry_bar = i
                    stop = entry - params.trail_atr * atr[i - 1]
            if active:
                close = closes[i]
                if (not np.isfinite(close) or close <= stop
                        or i - entry_bar + 1 >= params.hold_bars):
                    active = False
                else:
                    # Stops are evaluated at closes; each exit fills next open.
                    if close >= entry * (1.0 + params.breakeven_pct):
                        armed = True
                    if armed:
                        stop = max(stop, entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, close - params.trail_atr * atr[i])
                    decisions[i] = 1
            elif gate[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    p = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6, "high": [103.0] * 6,
        "low": [99.0] * 6, "close": [100.0, 101.0, 102.0, 101.0, 100.0, 100.0],
        "volume": [1000.0] * 6
    }, index=index)
    ind = pd.DataFrame({"entry_gate": [True, False, False, False, False, False],
                        "atr": [0.5] * 6}, index=index)
    result = strategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, ind, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()
