from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_window: int = 20
    noise_window: int = 20
    atr_window: int = 14
    entry_z: float = 1.4
    breakeven_pct: float = 0.006
    trail_atr: float = 1.25
    max_hold: int = 2
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791266212"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.trend_window + 1, params.noise_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.trend_window, params.noise_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        n = params.trend_window
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        t = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        out["log_sum"] = out["log_close"].rolling(n).sum()
        out["weighted_sum"] = (t * out["log_close"]).rolling(n).sum()
        out["level"] = out["log_sum"] / n
        out["slope"] = (out["weighted_sum"] - (t - (n - 1) / 2) * out["log_sum"]) / (n * (n * n - 1) / 12)
        out["prior_slope"] = out["slope"].shift(1)
        out["forecast"] = (out["level"] + out["slope"] * (n + 1) / 2).shift(1)
        out["return"] = out["log_close"].diff()
        out["noise"] = out["return"].shift(1).rolling(params.noise_window).std().replace(0, np.nan)
        out["innovation"] = out["return"] - out["prior_slope"]
        out["innovation_z"] = out["innovation"] / out["noise"]
        out["forecast_error"] = out["log_close"] - out["forecast"]
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if not (params.entry_z > 0 and params.trail_atr > 0 and params.breakeven_pct > 0
                and params.max_hold >= 1 and 0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Invalid entry, exit, holding, or size parameters.")
        z = indicators["innovation_z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        eligible = (
            (indicators["innovation_z"] <= -params.entry_z)
            & (indicators["prior_slope"] > 0)
            & (indicators["forecast_error"] < 0)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy()
        opening = data["open"].to_numpy(dtype=float)
        high = data["high"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        active = False
        entry_bar = -1
        entry = stop = peak = 0.0
        armed = False
        allocation = params.min_size
        for i in range(len(data)):
            if active:
                # The preceding close's entry decision fills at this open.
                if entry_bar < 0:
                    entry_bar = i
                    entry = opening[i]
                    peak = entry
                    stop = entry - params.trail_atr * atr[i - 1]
                    if not np.isfinite(entry) or entry <= 0:
                        active = False
                        continue
                valid = np.isfinite([high[i], low[i], close[i], atr[i]]).all()
                # Only the stop carried into this session can trigger today.
                # A trigger schedules liquidation at the following open.
                if not valid or low[i] <= stop or i - entry_bar + 1 >= params.max_hold:
                    active = False
                    entry_bar = -1
                    continue
                peak = max(peak, high[i])
                if peak >= entry * (1 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                signal[i] = 1
                size[i] = allocation
            elif eligible[i]:
                active = True
                entry_bar = -1
                armed = False
                # Conviction is fixed at entry; holding does not rebalance.
                strength = np.clip((-z[i] / params.entry_z - 1) / 2, 0, 1)
                allocation = float(params.min_size + (params.max_size - params.min_size) * strength)
                signal[i] = 1
                size[i] = allocation
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0] * 5, "high": [100.0, 102.0, 102.5, 103.0, 100.0],
        "low": [99.0, 99.0, 100.2, 100.1, 99.0],
        "close": [100.0, 101.0, 101.0, 102.0, 100.0], "volume": [1000.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "innovation_z": [-2.8, 0.0, 0.0, 0.0, 0.0],
        "prior_slope": [0.001] * 5, "forecast_error": [-0.01] * 5,
        "atr": [1.0] * 5,
    }, index=index)
    p = Params(trail_atr=2.0, max_hold=10)
    result = strategy.generate_signals(data, indicators, None, p).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert np.isclose(result["size"].iloc[1], 0.6)
    p.max_hold = 2
    assert strategy.generate_signals(data, indicators, None, p).data["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "low"] = 97.0
    assert strategy.generate_signals(data, indicators, None, p).data["signal"].tolist() == [0, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
