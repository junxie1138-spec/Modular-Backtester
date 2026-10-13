from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    match_tolerance: float = 0.45
    trail_atr: float = 0.8


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791245308"

    @classmethod
    def params_type(cls) -> type[AnalogParams]:
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return 280

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not np.isfinite(params.match_tolerance) or params.match_tolerance <= 0:
            raise ValueError("match_tolerance must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        tr = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = tr.rolling(14, min_periods=14).mean().replace(0, np.nan)
        out["drawdown"] = close.div(close.rolling(20, min_periods=20).max()).sub(1)
        scale = close.div(out["atr"])
        out["depth"] = -out["drawdown"] * scale
        out["dwell"] = -out["drawdown"].rolling(6, min_periods=6).mean() * scale
        out["relief"] = out["drawdown"].diff(5) * scale
        features = out[["depth", "dwell", "relief"]]
        distances = []
        outcomes = []
        # Each analog's next-open to third-open return has matured by this bar.
        for lag in range(3, 255):
            delta = features - features.shift(lag)
            distances.append(np.sqrt(delta.pow(2).mean(axis=1, skipna=False)).to_numpy())
            outcomes.append(data["open"].shift(lag - 3).div(
                data["open"].shift(lag - 1)).sub(1).to_numpy())
        distance = np.asarray(distances)
        outcome = np.asarray(outcomes)
        eligible = np.isfinite(distance) & np.isfinite(outcome)
        distance = np.where(eligible, distance, np.inf)
        nearest = np.argpartition(distance, 6, axis=0)[:7]
        selected_distance = np.take_along_axis(distance, nearest, axis=0)
        selected_return = np.take_along_axis(outcome, nearest, axis=0)
        valid = np.isfinite(selected_distance).all(axis=0)
        out["analog_distance"] = np.where(valid, selected_distance.max(axis=0), np.nan)
        out["forecast"] = np.where(valid, selected_return.mean(axis=0), np.nan)
        out["forecast_se"] = np.where(
            valid, selected_return.std(axis=0, ddof=1) / np.sqrt(7), np.nan)
        threshold = out["forecast_se"].clip(lower=0.0014)
        ready = (
            (out["analog_distance"] <= params.match_tolerance)
            & (out["depth"] >= 0.3)
            & (out["atr"] > 0)
        )
        out["candidate"] = np.select(
            [ready & (out["forecast"] > threshold),
             ready & (out["forecast"] < -threshold)], [1, -1], default=0)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("candidate")] = 0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AnalogParams
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signal = np.zeros(len(data), dtype=int)
        opening = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        side = 0
        entry_bar = -1
        entry = np.nan
        stop = -np.inf
        best = -np.inf
        activated = False
        for i in range(len(data)):
            if side:
                if entry_bar == -1:
                    # Yesterday's decision fills at today's open.
                    entry = opening[i]
                    entry_bar = i
                    best = side * entry
                    stop = best - params.trail_atr * atr[i - 1]
                if not np.isfinite(entry) or entry <= 0 or not np.isfinite(close[i]) or not np.isfinite(atr[i]):
                    side = 0
                else:
                    # Signed prices make the stop ratchet upward for either direction.
                    price = side * close[i]
                    best = max(best, price)
                    activated = activated or best - side * entry >= 0.004 * entry
                    if activated:
                        stop = max(stop, side * entry, best - params.trail_atr * atr[i])
                    # Stops are evaluated at closes; execution is next open.
                    if price <= stop or i - entry_bar + 1 >= 2:
                        side = 0
            elif candidate[i] and i >= self.warmup_bars(params) and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidate[i])
                entry_bar = -1
                activated = False
            signal[i] = side
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    n = 350
    t = np.arange(n, dtype=float)
    close = 100 + 0.015 * t + 2 * np.sin(t / 5)
    data = pd.DataFrame(
        {"open": close + 0.1, "high": close + 0.8, "low": close - 0.8,
         "close": close, "volume": np.full(n, 1000000.0)},
        index=pd.date_range("2020-01-01", periods=n, freq="B"))
    strategy = GeneratedStrategy()
    params = AnalogParams()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-10], params)
    pd.testing.assert_frame_equal(full.iloc[:-10], prefix)
    signals = strategy.generate_signals(data, full, None, params).data
    short = strategy.generate_signals(data.iloc[:-10], prefix, None, params).data
    pd.testing.assert_frame_equal(signals.iloc[:-10], short)
    forced = full.copy()
    forced["candidate"] = 0
    forced.iloc[280, forced.columns.get_loc("candidate")] = 1
    forced.iloc[290, forced.columns.get_loc("candidate")] = -1
    checked = strategy.generate_signals(data, forced, None, params).data["signal"]
    assert checked.iloc[280] == 0 and checked.iloc[281] == 1
    assert checked.iloc[283] == 0 and checked.iloc[291] == -1
    assert checked.iloc[293] == 0
    assert signals["signal"].isin([-1, 0, 1]).all()
    assert (signals["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
