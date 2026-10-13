from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRangeParams:
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapRangeParams]):
    strategy_id = "gen_local-w2_1791266422"

    @classmethod
    def params_type(cls):
        return GapRangeParams

    def warmup_bars(self, params: GapRangeParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: GapRangeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        ).replace([np.inf, -np.inf], np.nan)
        out["prior_range_mean"] = out["range"].rolling(20, min_periods=20).mean().shift(1)
        out["expanded"] = out["range"] > 1.5 * out["prior_range_mean"]
        out["up_gap_rejected"] = (
            (data["open"] > data["high"].shift(1))
            & (data["close"] < data["high"].shift(1))
            & (data["close"] < data["open"])
            & out["expanded"]
        ).fillna(False)
        out["down_gap_rejected"] = (
            (data["open"] < data["low"].shift(1))
            & (data["close"] > data["low"].shift(1))
            & (data["close"] > data["open"])
            & out["expanded"]
        ).fillna(False)
        out["previous_midpoint"] = (data["high"].shift(1) + data["low"].shift(1)) / 2.0
        out["contained_contraction"] = (
            (out["range"] < 0.65 * out["range"].shift(1))
            & (data["high"] <= data["high"].shift(1))
            & (data["low"] >= data["low"].shift(1))
        ).fillna(False)
        out["long_entry"] = (
            out["down_gap_rejected"].shift(1, fill_value=False)
            & out["contained_contraction"]
            & (data["close"] > out["previous_midpoint"])
        ).fillna(False)
        out["short_entry"] = (
            out["up_gap_rejected"].shift(1, fill_value=False)
            & out["contained_contraction"]
            & (data["close"] < out["previous_midpoint"])
        ).fillna(False)
        out["candidate"] = np.select(
            [out["long_entry"], out["short_entry"]], [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRangeParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if not isinstance(params.max_hold_bars, int) or params.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be a positive integer")
        decisions = np.zeros(len(data), dtype=int)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side, entry_bar, entry_price = 0, -1, 0.0
        for i in range(len(data)):
            if side:
                if entry_bar < 0:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    entry_bar, entry_price = i, opens[i]
                gain = side * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                expired = i - entry_bar + 1 >= params.max_hold_bars
                # Exits are close decisions, executed at the next open.
                if target_hit or expired:
                    side, entry_bar, entry_price = 0, -1, 0.0
                    continue
            elif candidates[i] in (-1, 1):
                side = int(candidates[i])
                entry_bar = -1
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=25, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    for direction in (1, -1):
        candidate = pd.DataFrame({"candidate": [direction] + [0] * 24}, index=index)
        result = strategy.generate_signals(data, candidate, None, GapRangeParams()).data
        assert result["signal"].iloc[0] == 0
        assert (result["signal"].iloc[1:19] == direction).all()
        assert (result["signal"].iloc[19:] == 0).all()
        moved = data.copy()
        moved.iloc[2, moved.columns.get_loc("close")] = 105.0 if direction == 1 else 95.0
        result = strategy.generate_signals(moved, candidate, None, GapRangeParams()).data
        assert result["signal"].iloc[2] == direction
        assert result["signal"].iloc[3] == 0
    data.iloc[20] = [103.0, 104.0, 99.0, 100.0, 1000.0]
    data.iloc[21] = [101.0, 102.0, 100.0, 100.5, 1000.0]
    derived = strategy.indicators(data, GapRangeParams())
    assert derived["candidate"].iloc[21] == -1
    prefix = strategy.indicators(data.iloc[:22], GapRangeParams())
    pd.testing.assert_frame_equal(prefix, derived.iloc[:22])


if __name__ == "__main__":
    _self_check()
