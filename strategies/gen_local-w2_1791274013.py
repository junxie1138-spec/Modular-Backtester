from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    atr_bars: int = 14
    min_gap_atr: float = 0.45
    max_adverse_fraction: float = 0.25
    trail_atr: float = 0.75
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local-w2_1791274013"

    @classmethod
    def params_type(cls):
        return GapRepairParams

    def warmup_bars(self, params: GapRepairParams) -> int:
        return int(max(200, params.atr_bars + 2))

    def indicators(self, data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_bars, min_periods=params.atr_bars
        ).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["gap_points"] = out["prior_close"] - data["open"]
        downside_gap = out["gap_points"].where(out["gap_points"] > 0)
        out["gap_atr"] = downside_gap / out["prior_atr"].where(out["prior_atr"] > 0)
        out["repair_fraction"] = (data["close"] - data["open"]) / downside_gap
        out["adverse_fraction"] = (data["open"] - data["low"]) / downside_gap
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRepairParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        prior_close = indicators["prior_close"].to_numpy(dtype=float)
        entry = (
            (data["close"] > indicators["ma200"])
            & (indicators["gap_atr"] >= params.min_gap_atr)
            & indicators["repair_fraction"].between(0.20, 0.65)
            & indicators["adverse_fraction"].between(0.0, params.max_adverse_fraction)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = np.nan
        stop = -np.inf
        target = np.nan

        for i in range(len(data)):
            if active:
                held += 1
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                # The first in-trade close occurs after the next-open fill.
                high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or close[i] >= target or held >= params.max_hold_bars:
                    active = False
                else:
                    signals[i] = 1
            elif entry[i] and np.isfinite(close[i]) and np.isfinite(prior_close[i]):
                active = True
                held = 0
                high_water = np.nan
                stop = -np.inf
                target = prior_close[i]
                signals[i] = 1

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 101.0, 100.0, 99.0, 99.0]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 10.0, 10.0, 10.0],
        "prior_close": [110.0] * 5,
        "ma200": [90.0] * 5,
        "gap_atr": [0.8, np.nan, np.nan, np.nan, np.nan],
        "repair_fraction": [0.4] * 5,
        "adverse_fraction": [0.1] * 5,
    }, index=index)
    params = GapRepairParams(max_hold_bars=4)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    # ATR expansion cannot lower the stop; the exit decision fills one bar later.
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    params.max_hold_bars = 1
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
