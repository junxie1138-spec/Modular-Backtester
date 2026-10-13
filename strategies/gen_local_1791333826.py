from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMidpointParams:
    min_gap_atr: float = 0.5
    trailing_atr: float = 1.8


class GeneratedStrategy(BaseStrategy[GapMidpointParams]):
    strategy_id = "gen_local_1791333826"

    @classmethod
    def params_type(cls) -> type[GapMidpointParams]:
        return GapMidpointParams

    @staticmethod
    def warmup_bars(params: GapMidpointParams) -> int:
        return 21

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapMidpointParams) -> pd.DataFrame:
        if not np.isfinite(params.min_gap_atr) or params.min_gap_atr <= 0:
            raise ValueError("min_gap_atr must be finite and positive")
        if not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0:
            raise ValueError("trailing_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["range_high"] = data["high"].rolling(20, min_periods=20).max().shift(1)
        out["range_low"] = data["low"].rolling(20, min_periods=20).min().shift(1)
        out["midpoint"] = (out["range_high"] + out["range_low"]) / 2.0
        out["gap_atr"] = (data["open"] - out["prior_close"]) / out["atr"].where(out["atr"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMidpointParams,
    ) -> SignalFrame:
        valid = (
            np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
            & np.isfinite(indicators).all(axis=1)
            & indicators["atr"].gt(0)
            & indicators["range_high"].gt(indicators["range_low"])
        )
        setup = (
            valid
            & indicators["gap_atr"].le(-params.min_gap_atr)
            & indicators["prior_close"].gt(indicators["midpoint"])
            & data["open"].lt(indicators["midpoint"])
            & data["close"].gt(indicators["midpoint"])
            & data["close"].lt(indicators["prior_close"])
        ).fillna(False).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = valid.to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if not ready[i]:
                active = False
                age = 0
                high_water = stop = -np.inf
                continue
            if active:
                # Entry fills on the bar after setup; track only in-trade closes.
                age += 1
                high_water = max(high_water, closes[i])
                stop = max(stop, high_water - params.trailing_atr * atr[i])
                if closes[i] <= stop or age >= 4:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
                else:
                    signals[i] = 1
            elif setup[i]:
                active = True
                age = 0
                high_water = stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    data = pd.DataFrame({
        "open": np.full(32, 101.0),
        "high": np.full(32, 102.0),
        "low": np.full(32, 98.0),
        "close": np.full(32, 101.0),
        "volume": np.full(32, 1000000.0),
    }, index=pd.date_range("2020-01-01", periods=32, freq="B"))
    data.iloc[21, data.columns.get_loc("open")] = 98.0
    data.iloc[21, data.columns.get_loc("close")] = 100.5
    strategy = GeneratedStrategy()
    params = GapMidpointParams()
    result = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert result["signal"].iloc[21:27].tolist() == [0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()
    data.iloc[23, data.columns.get_loc("close")] = 95.0
    data.iloc[23, data.columns.get_loc("low")] = 94.0
    params = GapMidpointParams(trailing_atr=0.5)
    result = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert result["signal"].iloc[21:26].tolist() == [0, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
