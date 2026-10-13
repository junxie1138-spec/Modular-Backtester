from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    cluster_window: int = 5
    volume_window: int = 20
    atr_window: int = 14
    gap_atr: float = 0.15
    volume_floor: float = 1.2
    capacity: float = 3.2
    breakeven_pct: float = 0.02
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_local_1790896341"

    @classmethod
    def params_type(cls):
        return GapCapacityParams

    def warmup_bars(self, params: GapCapacityParams) -> int:
        return int(max(200, params.volume_window + params.cluster_window,
                       params.atr_window + params.cluster_window + 1))

    def indicators(self, data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        if min(params.cluster_window, params.volume_window, params.atr_window,
               params.hold_bars) < 1 or min(params.gap_atr, params.volume_floor,
               params.capacity, params.breakeven_pct, params.trail_atr) <= 0:
            raise ValueError("Lookbacks and thresholds must be positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["volume_baseline"] = data["volume"].rolling(
            params.volume_window, min_periods=params.volume_window).mean().shift(1)
        out["relative_volume"] = data["volume"] / out["volume_baseline"].replace(0, np.nan)
        out["gap_units"] = (data["open"] - out["previous_close"]) / out["prior_atr"].replace(0, np.nan)
        out["up_gap"] = (
            (out["gap_units"] >= params.gap_atr)
            & (data["low"] > out["previous_close"])
            & (data["close"] >= data["open"])
            & (out["relative_volume"] >= params.volume_floor)
        ).fillna(False)
        out["down_gap"] = (
            (out["gap_units"] <= -params.gap_atr)
            & (data["high"] < out["previous_close"])
            & (data["close"] <= data["open"])
            & (out["relative_volume"] >= params.volume_floor)
        ).fillna(False)
        out["gap_load"] = (
            out["up_gap"].astype(int) - out["down_gap"].astype(int)
        ) * out["relative_volume"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["pressure"] = out["gap_load"].rolling(
            params.cluster_window, min_periods=params.cluster_window).sum()
        out["up_count"] = out["up_gap"].rolling(
            params.cluster_window, min_periods=params.cluster_window).sum()
        out["down_count"] = out["down_gap"].rolling(
            params.cluster_window, min_periods=params.cluster_window).sum()
        out["ready"] = (
            np.isfinite(out[["atr", "ma200", "pressure", "relative_volume"]]).all(axis=1)
            & np.isfinite(data[["open", "high", "low", "close", "volume"]]).all(axis=1)
            & (out["atr"] > 0) & (data["low"] > 0) & (data["volume"] >= 0)
        )
        out["long_entry"] = (
            out["ready"] & out["up_gap"] & (out["up_count"] >= 2)
            & (out["pressure"] >= params.capacity)
            & (data["close"] > out["ma200"])
        )
        out["short_entry"] = (
            out["ready"] & out["down_gap"] & (out["down_count"] >= 2)
            & (out["pressure"] <= -params.capacity)
            & (data["close"] < out["ma200"])
        )
        out["setup"] = out["long_entry"].astype(int) - out["short_entry"].astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapCapacityParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        direction, age = 0, 0
        entry = stop = entry_atr = np.nan
        armed = False
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        setup = indicators["setup"].fillna(0).to_numpy(dtype=int)
        for i in range(len(data)):
            if not ready[i]:
                direction = 0
                entry = np.nan
                continue
            if direction:
                # The prior close's decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                    stop = direction * entry - params.trail_atr * entry_atr
                age += 1
                signed_close = direction * closes[i]
                signed_entry = direction * entry
                exited = signed_close <= stop or age >= params.hold_bars
                if not exited:
                    armed = armed or signed_close - signed_entry >= entry * params.breakeven_pct
                    if armed:
                        # Signed-price stops only rise; actual short stops only fall.
                        stop = max(stop, signed_entry,
                                   signed_close - params.trail_atr * atr[i])
                    exited = signed_close <= stop
                # Stops are evaluated at the close; exits fill next bar.
                if exited:
                    direction = 0
                    entry = np.nan
                signals[i] = direction
                continue
            if setup[i]:
                direction = int(setup[i])
                age, armed = 0, False
                entry, entry_atr = np.nan, atr[i]
                signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapCapacityParams()
    index = pd.date_range("2020-01-01", periods=6)
    ctx = StrategyContext("SPY", "1d", 0)
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0.0, 3.0, 1.0, -1.0, 0.0, 0.0])
        data = pd.DataFrame({"open": 100.0, "close": close}, index=index)
        ind = pd.DataFrame({"atr": 2.0, "ready": True,
                            "setup": [direction, 0, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0, 0]
        assert (result["size"] > 0).all()
        prefix = strategy.generate_signals(data.iloc[:3], ind.iloc[:3], ctx, params).data
        assert prefix.equals(result.iloc[:3])
        params.hold_bars = 2
        timed = strategy.generate_signals(data, ind, ctx, params).data
        assert timed["signal"].tolist() == [0, direction, direction, 0, 0, 0]
        params.hold_bars = 18


if __name__ == "__main__":
    _self_check()
