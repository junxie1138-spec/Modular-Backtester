from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    lookback: int = 20
    gap_atr: float = 0.5
    max_volume_ratio: float = 0.9
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local_1791285589"

    @classmethod
    def params_type(cls) -> type[GapRepairParams]:
        return GapRepairParams

    def warmup_bars(self, params: GapRepairParams) -> int:
        return int(max(200, params.lookback + 2))

    def indicators(self, data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        if params.lookback < 2 or params.gap_atr <= 0 or params.max_volume_ratio <= 0:
            raise ValueError("Lookback must be >= 2 and thresholds must be positive.")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["prior_close"]).abs(),
             (data["low"] - out["prior_close"]).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["prior_atr"] = out["true_range"].rolling(
            params.lookback, min_periods=params.lookback
        ).mean().shift(1)
        out["prior_volume"] = data["volume"].rolling(
            params.lookback, min_periods=params.lookback
        ).mean().shift(1)
        out["volume_ratio"] = data["volume"] / out["prior_volume"].where(
            out["prior_volume"] > 0
        )
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        valid = (
            np.isfinite(out).all(axis=1)
            & np.isfinite(data).all(axis=1)
            & (out["prior_atr"] > 0)
            & (data["volume"] > 0)
        )
        quiet = out["volume_ratio"] <= params.max_volume_ratio
        out["long_setup"] = (
            valid & quiet
            & (out["gap"] <= -params.gap_atr * out["prior_atr"])
            & (data["open"] < out["prior_low"])
            & (data["close"] >= out["prior_close"])
            & (data["close"] > out["ma200"])
        ).fillna(False)
        out["short_setup"] = (
            valid & quiet
            & (out["gap"] >= params.gap_atr * out["prior_atr"])
            & (data["open"] > out["prior_high"])
            & (data["close"] <= out["prior_close"])
            & (data["close"] < out["ma200"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapRepairParams
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        entries = np.where(
            indicators["long_setup"].fillna(False), 1,
            np.where(indicators["short_setup"].fillna(False), -1, 0)
        )
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if position and bar == exit_bar:
                position = 0
            elif position == 0 and entry:
                position = int(entry)
                exit_bar = bar + params.hold_bars
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapRepairParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=225, freq="B")
    for direction in (1, -1):
        close = 100.0 + direction * 0.1 * np.arange(len(index))
        data = pd.DataFrame(
            {"open": close, "high": close + 1.0, "low": close - 1.0,
             "close": close, "volume": 1000.0}, index=index
        )
        data.loc[index[210], "open"] = close[209] - direction * 2.0
        data.loc[index[210], "high"] = max(close[210], data.loc[index[210], "open"]) + 1.0
        data.loc[index[210], "low"] = min(close[210], data.loc[index[210], "open"]) - 1.0
        data.loc[index[210], "volume"] = 500.0
        indicators = strategy.indicators(data, params)
        indicators.loc[index[212], ["long_setup", "short_setup"]] = [direction < 0, direction > 0]
        result = strategy.generate_signals(data, indicators, None, params).data
        expected = np.zeros(len(index), dtype=int)
        expected[211:211 + params.hold_bars] = direction
        assert np.array_equal(result["signal"].to_numpy(), expected)
        assert result.index.equals(data.index) and (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
