from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEchoParams:
    corr_window: int = 48
    horizon: int = 4
    threshold: float = 0.08


class GeneratedStrategy(BaseStrategy[RangeEchoParams]):
    strategy_id = "gen_local-w2_1791319819"

    @classmethod
    def params_type(cls) -> type[RangeEchoParams]:
        return RangeEchoParams

    @staticmethod
    def warmup_bars(params: RangeEchoParams) -> int:
        return int(params.corr_window + 2 * params.horizon)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeEchoParams) -> pd.DataFrame:
        if params.corr_window < 3 or params.horizon < 1:
            raise ValueError("corr_window must be >= 3 and horizon >= 1")
        if not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["previous_range"] = out["range"].shift(1)
        out["upper_extension"] = data["high"].diff().clip(lower=0)
        out["lower_extension"] = (-data["low"].diff()).clip(lower=0)
        out["range_impulse"] = (
            (out["upper_extension"] - out["lower_extension"])
            / out["previous_range"]
        )
        out["range_impulse"] = out["range_impulse"].replace([np.inf, -np.inf], np.nan)
        out["block_impulse"] = out["range_impulse"].rolling(
            params.horizon, min_periods=params.horizon
        ).mean()
        out["previous_block"] = out["block_impulse"].shift(params.horizon)
        out["range_autocorrelation"] = out["block_impulse"].rolling(
            params.corr_window, min_periods=params.corr_window
        ).corr(out["previous_block"]).clip(-1.0, 1.0)
        out["range_forecast"] = out["range_autocorrelation"] * out["block_impulse"]
        out["candle_pressure"] = (data["close"] - data["open"]) / out["range"]
        out["pressure"] = out["candle_pressure"].rolling(
            params.horizon, min_periods=params.horizon
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeEchoParams,
    ) -> SignalFrame:
        forecast = indicators["range_forecast"]
        pressure = indicators["pressure"]
        valid = np.isfinite(forecast) & np.isfinite(pressure)
        long_entry = valid & (forecast > params.threshold) & (pressure > params.threshold)
        short_entry = valid & (forecast < -params.threshold) & (pressure < -params.threshold)
        agreement = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(agreement):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Opposite joint agreement exits; reentry requires a later bar.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RangeEchoParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    test_indicators = pd.DataFrame({
        "range_forecast": [0.2, 0.2, np.nan, -0.2, -0.2, 0.0, 0.2, 0.2],
        "pressure": [0.2, -0.2, 0.2, -0.2, -0.2, 0.0, 0.2, 0.2],
    }, index=index)
    result = strategy.generate_signals(data, test_indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert (result["size"] > 0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1000.0},
                        index=pd.date_range("2020-01-01", periods=80))
    derived = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, derived, None, params).data["signal"].eq(0).all()
