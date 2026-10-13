from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRangeParams:
    range_window: int = 20
    min_gap_ranges: float = 0.4
    contraction_ratio: float = 0.85
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapRangeParams]):
    strategy_id = "gen_local_1791268320"

    @classmethod
    def params_type(cls) -> type[GapRangeParams]:
        return GapRangeParams

    @staticmethod
    def warmup_bars(params: GapRangeParams) -> int:
        return int(params.range_window + 3)

    def indicators(self, data: pd.DataFrame, params: GapRangeParams) -> pd.DataFrame:
        if params.range_window < 2 or not 3 <= params.hold_bars <= 5:
            raise ValueError("range_window must be >= 2 and hold_bars must be 3-5")
        if not np.isfinite(params.min_gap_ranges) or params.min_gap_ranges <= 0:
            raise ValueError("min_gap_ranges must be finite and positive")
        if not 0 < params.contraction_ratio < 1:
            raise ValueError("contraction_ratio must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["baseline_range"] = out["range"].rolling(
            params.range_window, min_periods=params.range_window
        ).mean().shift(1).replace(0.0, np.nan)
        out["up_gap_ranges"] = (
            data["open"] - out["previous_high"]
        ) / out["baseline_range"]
        out["down_gap_ranges"] = (
            out["previous_low"] - data["open"]
        ) / out["baseline_range"]
        out["up_event"] = (
            out["up_gap_ranges"].ge(params.min_gap_ranges)
            & data["low"].gt(out["previous_high"])
        ).fillna(False)
        out["down_event"] = (
            out["down_gap_ranges"].ge(params.min_gap_ranges)
            & data["high"].lt(out["previous_low"])
        ).fillna(False)
        out["contracting"] = (
            out["range"].gt(0)
            & out["range"].le(out["range"].shift(1) * params.contraction_ratio)
        ).fillna(False)
        out["retreat_down"] = out["midpoint"].lt(out["midpoint"].shift(1)).fillna(False)
        out["retreat_up"] = out["midpoint"].gt(out["midpoint"].shift(1)).fillna(False)
        out["up_anchor"] = out["previous_high"].shift(2)
        out["down_anchor"] = out["previous_low"].shift(2)
        out["two_contractions"] = (
            out["contracting"] & out["contracting"].shift(1, fill_value=False)
        )
        out["short_entry"] = (
            out["up_event"].shift(2, fill_value=False)
            & out["two_contractions"]
            & out["retreat_down"]
            & out["retreat_down"].shift(1, fill_value=False)
            & data["low"].shift(1).gt(out["up_anchor"])
            & data["low"].gt(out["up_anchor"])
        ).fillna(False)
        out["long_entry"] = (
            out["down_event"].shift(2, fill_value=False)
            & out["two_contractions"]
            & out["retreat_up"]
            & out["retreat_up"].shift(1, fill_value=False)
            & data["high"].shift(1).lt(out["down_anchor"])
            & data["high"].lt(out["down_anchor"])
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
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be 3-5")
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapRangeParams()
    strategy = GeneratedStrategy()
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=32),
    )
    data.iloc[20:23, :4] = np.array([
        [106.0, 108.0, 102.0, 104.0],
        [104.0, 106.0, 102.0, 103.0],
        [103.0, 104.0, 102.0, 103.0],
    ])
    for direction in (-1, 1):
        sample = data.copy()
        if direction == 1:
            sample["open"] = 200.0 - data["open"]
            sample["high"] = 200.0 - data["low"]
            sample["low"] = 200.0 - data["high"]
            sample["close"] = 200.0 - data["close"]
        derived = strategy.indicators(sample, params)
        assert derived["candidate"].iloc[:22].eq(0).all()
        assert derived["candidate"].iloc[22] == direction
        # Opposite candidates during a trade must not change its fixed exit.
        derived.loc[sample.index[23:27], "candidate"] = -direction
        result = strategy.generate_signals(sample, derived, None, params).data
        assert result["signal"].iloc[:23].eq(0).all()
        assert result["signal"].iloc[23:27].eq(direction).all()
        assert result["signal"].iloc[27] == 0
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
