from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    baseline_window: int = 20
    rank_window: int = 60
    tail_percentile: float = 0.90
    breakeven_pct: float = 0.004
    trail_atr: float = 0.7


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790962006"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(200, params.baseline_window + params.rank_window + 3, 15)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.baseline_window < 2 or params.rank_window < 3:
            raise ValueError("Windows must be at least 2 and 3 respectively.")
        if not 0.7 < params.tail_percentile < 1.0:
            raise ValueError("tail_percentile must lie between 0.7 and 1.")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError("Exit parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).replace(0, np.nan)
        out["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]
        out["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
        out["balance"] = (out["lower_wick"] - out["upper_wick"]) / out["range"]
        out["recent_balance"] = out["balance"].rolling(2, min_periods=2).mean()
        out["baseline"] = out["balance"].rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean().shift(2)
        out["breakpoint"] = out["recent_balance"] - out["baseline"]
        out["percentile"] = out["breakpoint"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["previous_percentile"] = out["percentile"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        valid = indicators[[
            "percentile", "previous_percentile", "ma200", "atr",
            "breakpoint", "recent_balance"
        ]].notna().all(axis=1) & indicators["atr"].gt(0)
        long_entry = (
            valid & data["close"].gt(indicators["ma200"])
            & indicators["percentile"].ge(params.tail_percentile)
            & indicators["previous_percentile"].le(0.7)
            & indicators["breakpoint"].gt(0)
            & indicators["recent_balance"].gt(0)
        )
        short_entry = (
            valid & data["close"].lt(indicators["ma200"])
            & indicators["percentile"].le(1.0 - params.tail_percentile)
            & indicators["previous_percentile"].ge(0.3)
            & indicators["breakpoint"].lt(0)
            & indicators["recent_balance"].lt(0)
        )
        candidates = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry = best = stop = np.nan
        armed = False
        for i, (opening, high, low, closing) in enumerate(prices):
            if side:
                if np.isnan(entry):
                    entry = side * opening
                    best = entry
                    stop = entry - 2.0 * atr[i - 1]
                    age = 0
                    armed = False
                age += 1
                favorable = high if side == 1 else -low
                adverse = low if side == 1 else -high
                # Existing stop is checked before using this bar's new extreme.
                exited = adverse <= stop
                if not exited:
                    best = max(best, favorable)
                    if best - entry >= abs(entry) * params.breakeven_pct:
                        armed = True
                    if armed:
                        # Signed coordinates make both directions ratchet upward.
                        stop = max(stop, entry, best - params.trail_atr * atr[i])
                    exited = side * closing <= stop or age >= 2
                if exited:
                    side = 0
                    entry = np.nan
            elif candidates[i]:
                side = int(candidates[i])
                entry = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        # Close decisions execute at the following open, including stop exits.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6, "high": [100.1] * 6,
        "low": [99.9] * 6, "close": [100.0] * 6,
        "volume": [1000.0] * 6,
    }, index=index)
    ind = pd.DataFrame({
        "percentile": [0.95, 0.5, 0.5, 0.05, 0.5, 0.5],
        "previous_percentile": [0.5] * 6,
        "ma200": [99.0] * 3 + [101.0] * 3,
        "atr": [1.0] * 6, "breakpoint": [1.0] * 3 + [-1.0] * 3,
        "recent_balance": [1.0] * 3 + [-1.0] * 3,
    }, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    data.loc[index[1], ["high", "close"]] = [101.0, 100.1]
    data.loc[index[4], ["low", "close"]] = [99.0, 99.9]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0]
