from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRangeParams:
    expansion_multiple: float = 1.5
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[GapRangeParams]):
    strategy_id = "gen_local_1790906502"

    @classmethod
    def params_type(cls):
        return GapRangeParams

    def warmup_bars(self, params: GapRangeParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: GapRangeParams) -> pd.DataFrame:
        if not np.isfinite(params.expansion_multiple) or params.expansion_multiple <= 1:
            raise ValueError("expansion_multiple must be finite and greater than one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["range_baseline"] = out["range"].rolling(20, min_periods=20).median().shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["gap_atr"] = (data["open"] - out["previous_close"]) / out["atr"].shift(1).replace(0, np.nan)
        out["range_expansion"] = out["range"] / out["range_baseline"].replace(0, np.nan)
        out["inside_day"] = (data["high"] <= data["high"].shift(1)) & (data["low"] >= data["low"].shift(1))
        out["contracting"] = out["range"] <= 0.6 * out["range"].shift(1)
        out["gap_direction"] = np.sign(out["gap_atr"]).fillna(0).astype(int)
        out["retreat"] = (data["close"] - out["previous_close"]) * out["gap_direction"].shift(1) < 0
        out["setup"] = (
            (out["gap_atr"].shift(1).abs() >= 0.5)
            & (out["range_expansion"].shift(1) >= params.expansion_multiple)
            & out["inside_day"]
            & out["contracting"]
            & out["retreat"]
            & np.isfinite(out["atr"])
            & (out["atr"] > 0)
        ).fillna(False)
        out["candidate"] = (-out["gap_direction"].shift(1)).where(out["setup"], 0).fillna(0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRangeParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        side, held = 0, 0
        entry, frozen_atr = np.nan, np.nan
        for i in range(len(data)):
            if side:
                # The preceding close's decision fills at this bar's open.
                if np.isnan(entry):
                    entry = opens[i]
                held += 1
                # Freeze both entry and ATR; the stop never trails.
                stopped = side * (closes[i] - entry) < -params.stop_atr * frozen_atr
                if stopped or held >= 8:
                    side, held = 0, 0
                    entry = np.nan
                decisions[i] = side
                continue
            if candidates[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidates[i])
                frozen_atr = atr[i]
                entry = np.nan
                held = 0
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=20)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 94.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[2], "close"] = 95.0
    data.loc[index[6], "close"] = 105.0
    indicators = pd.DataFrame({"candidate": 0, "atr": 2.0}, index=index)
    indicators.loc[index[[0, 4, 8]], "candidate"] = [1, -1, 1]
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=22)
    result = strategy.generate_signals(data, indicators, ctx, GapRangeParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
