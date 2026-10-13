from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    hold_bars: int = 4
    max_distance: float = 0.20
    min_forecast: float = 0.003


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local-w2_1791272777"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not np.isfinite(params.max_distance) or params.max_distance < 0:
            raise ValueError("max_distance must be finite and nonnegative")
        if not np.isfinite(params.min_forecast) or params.min_forecast <= 0:
            raise ValueError("min_forecast must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["return"] = out["log_close"].diff()
        for lag in range(4):
            out[f"r{lag}"] = out["return"].shift(lag)
        a, b, c, d = (out[f"r{lag}"] for lag in range(4))
        out["fine_energy"] = ((a - b) ** 2 + (c - d) ** 2) / 2.0
        out["coarse_energy"] = (a + b - c - d) ** 2 / 4.0
        out["drift_energy"] = (a + b + c + d) ** 2 / 4.0
        out["total_energy"] = out[["fine_energy", "coarse_energy", "drift_energy"]].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 1e-16)
        features = []
        for scale in ("fine", "coarse", "drift"):
            name = f"{scale}_share"
            out[name] = out[f"{scale}_energy"] / denominator
            features.append(name)
        out["orientation"] = np.sign(a + b + c + d)
        # ponytail: only two matured analogs fit ten bars; expand the archive if warmup permits.
        for lag in (4, 5):
            delta = out[features] - out[features].shift(lag)
            out[f"distance_{lag}"] = delta.pow(2).sum(axis=1, min_count=3)
            out[f"outcome_{lag}"] = out["log_close"].shift(lag - 4) - out["log_close"].shift(lag)
            out[f"forecast_{lag}"] = (
                out[f"outcome_{lag}"] * out["orientation"].shift(lag) * out["orientation"]
            )
        ready = out[["distance_4", "distance_5", "forecast_4", "forecast_5"]].notna().all(axis=1)
        select_four = out["distance_4"] <= out["distance_5"]
        out["distance"] = out["distance_4"].where(select_four, out["distance_5"]).where(ready)
        out["forecast"] = out["forecast_4"].where(select_four, out["forecast_5"]).where(ready)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        forecast = indicators["forecast"]
        eligible = (
            np.isfinite(forecast)
            & np.isfinite(indicators["distance"])
            & (indicators["distance"] <= params.max_distance)
            & (forecast.abs() >= params.min_forecast)
        )
        entries = np.sign(forecast.where(eligible, 0.0)).fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        side = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if side != 0 and i == exit_bar:
                side = 0
                # Leave this decision bar flat, even if a fresh analog qualifies.
                continue
            if side == 0 and entry != 0:
                side = int(entry)
                exit_bar = i + params.hold_bars
            positions[i] = side
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    close = 100.0 * np.exp(np.cumsum(0.006 * np.sin(np.arange(40) * 1.7) + 0.001))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0},
        index=index,
    )
    params = AnalogParams()
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:30], strategy.indicators(data.iloc[:30], params))
    assert full["forecast"].iloc[:9].isna().all()
    for hold in (3, 4, 5):
        params = AnalogParams(hold_bars=hold)
        for side in (-1, 1):
            forced = pd.DataFrame({"forecast": side * 0.02, "distance": 0.0}, index=index)
            signal = strategy.generate_signals(data, forced, None, params).data["signal"]
            assert signal.iloc[0] == 0
            assert (signal.iloc[1 : hold + 1] == side).all()
            assert signal.iloc[hold + 1] == 0
            assert signal.iloc[hold + 2] == side


if __name__ == "__main__":
    _self_check()
