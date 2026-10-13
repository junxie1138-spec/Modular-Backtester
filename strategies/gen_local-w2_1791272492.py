from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapBalanceParams:
    rank_window: int = 60
    balance_tail: float = 0.20
    settlement_tail: float = 0.75
    hold_bars: int = 2


def _fixed_bar_positions(entries: np.ndarray, hold_bars: int) -> np.ndarray:
    positions = np.zeros(len(entries), dtype=int)
    position = 0
    remaining = 0
    for i, entry in enumerate(entries):
        if remaining > 0:
            positions[i] = position
            remaining -= 1
        elif position != 0:
            # Always emit a flat bar at expiry, even if another setup fires.
            position = 0
        elif entry != 0:
            position = int(entry)
            positions[i] = position
            remaining = hold_bars - 1
    return positions


class GeneratedStrategy(BaseStrategy[GapBalanceParams]):
    strategy_id = "gen_local-w2_1791272492"

    @classmethod
    def params_type(cls) -> type[GapBalanceParams]:
        return GapBalanceParams

    @staticmethod
    def _validate(params: GapBalanceParams) -> None:
        if params.rank_window < 2:
            raise ValueError("rank_window must be at least 2")
        if not 0.0 < params.balance_tail < 0.5:
            raise ValueError("balance_tail must be between 0 and 0.5")
        if not 0.5 < params.settlement_tail < 1.0:
            raise ValueError("settlement_tail must be between 0.5 and 1")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")

    def warmup_bars(self, params: GapBalanceParams) -> int:
        self._validate(params)
        # Cancellation uses the current and previous gap, each needing a prior close.
        return int(params.rank_window + 2)

    def indicators(self, data: pd.DataFrame, params: GapBalanceParams) -> pd.DataFrame:
        self._validate(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = data["open"].div(out["previous_close"]).sub(1)
        out["previous_gap"] = out["gap"].shift(1)
        out["gap_mass"] = out["gap"].abs() + out["previous_gap"].abs()
        out["cancellation_error"] = (out["gap"] + out["previous_gap"]).abs().div(
            out["gap_mass"].where(out["gap_mass"] > 0)
        )
        out["daily_range"] = data["high"] - data["low"]
        out["settlement_pressure"] = (2 * data["close"] - data["high"] - data["low"]).div(
            out["daily_range"].where(out["daily_range"] > 0)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        out["balance_rank"] = out["cancellation_error"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["settlement_rank"] = out["settlement_pressure"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapBalanceParams,
    ) -> SignalFrame:
        self._validate(params)
        x = indicators
        balanced = x["balance_rank"].le(params.balance_tail)
        long_setup = (
            balanced
            & x["gap"].gt(0)
            & x["previous_gap"].lt(0)
            & x["settlement_pressure"].gt(0)
            & x["settlement_rank"].ge(params.settlement_tail)
        )
        short_setup = (
            balanced
            & x["gap"].lt(0)
            & x["previous_gap"].gt(0)
            & x["settlement_pressure"].lt(0)
            & x["settlement_rank"].le(1.0 - params.settlement_tail)
        )
        required = ["gap", "previous_gap", "balance_rank", "settlement_pressure", "settlement_rank"]
        valid = x[required].notna().all(axis=1)
        entries = np.select(
            [long_setup.fillna(False) & valid, short_setup.fillna(False) & valid],
            [1, -1], default=0,
        ).astype(int)
        df = pd.DataFrame(index=data.index)
        df["signal"] = _fixed_bar_positions(entries, params.hold_bars)
        # Entry and fixed expiry both execute at the following bar's open.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    entries = np.array([1, -1, -1, -1, 1, 1], dtype=int)
    two_bar = _fixed_bar_positions(entries, 2)
    assert two_bar.tolist() == [1, 1, 0, -1, -1, 0]
    assert _fixed_bar_positions(entries, 1).tolist() == [1, 0, -1, 0, 1, 0]
    shifted = pd.Series(two_bar).shift(1).fillna(0).astype(int)
    assert shifted.tolist() == [0, 1, 1, 0, -1, -1]
