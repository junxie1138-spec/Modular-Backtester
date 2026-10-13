from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCapacityParams:
    window: int = 8
    gap_capacity: float = 0.8
    range_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[GapCapacityParams]):
    strategy_id = "gen_local_1790867925"

    @classmethod
    def params_type(cls) -> type[GapCapacityParams]:
        return GapCapacityParams

    def warmup_bars(self, params: GapCapacityParams) -> int:
        return int(params.window) + 1

    def indicators(self, data: pd.DataFrame, params: GapCapacityParams) -> pd.DataFrame:
        if params.window < 2 or params.gap_capacity <= 0 or params.range_threshold <= 0:
            raise ValueError("Require window >= 2 and positive thresholds.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["prior_range"] = out["range"].shift(1).replace(0, np.nan)
        out["prior_close"] = data["close"].shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        retained_up = (np.minimum(data["open"], data["close"]) - out["prior_close"]).clip(lower=0)
        retained_down = (np.maximum(data["open"], data["close"]) - out["prior_close"]).clip(upper=0)
        out["retained_up"] = retained_up
        out["retained_down"] = retained_down
        out["retained_gap"] = np.where(
            out["gap"] > 0, retained_up,
            np.where(out["gap"] < 0, retained_down, 0.0),
        )
        out["gap_units"] = (out["retained_gap"] / out["prior_range"]).replace([np.inf, -np.inf], np.nan)
        out["gap_load"] = out["gap_units"].rolling(params.window, min_periods=params.window).sum()
        out["upper_extension"] = (data["high"] - data["high"].shift(1)).clip(lower=0)
        out["lower_extension"] = (data["low"].shift(1) - data["low"]).clip(lower=0)
        out["edge_balance"] = out["upper_extension"] - out["lower_extension"]
        out["edge_sum"] = out["edge_balance"].rolling(params.window, min_periods=params.window).sum()
        out["range_sum"] = out["range"].rolling(params.window, min_periods=params.window).sum()
        out["range_pressure"] = (out["edge_sum"] / out["range_sum"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapCapacityParams,
    ) -> SignalFrame:
        long_entry = (
            (indicators["gap_load"] >= params.gap_capacity)
            & (indicators["range_pressure"] >= params.range_threshold)
        ).fillna(False)
        short_entry = (
            (indicators["gap_load"] <= -params.gap_capacity)
            & (indicators["range_pressure"] <= -params.range_threshold)
        ).fillna(False)
        qualification = np.where(long_entry, 1, np.where(short_entry, -1, 0))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, direction in enumerate(qualification):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Opposite AND qualification exits to flat before any new entry.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapCapacityParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    directions = np.array([1, 0, -1, -1, 0, 1, 1, 0])
    evidence = pd.DataFrame({
        "gap_load": directions * 2.0,
        "range_pressure": directions * 0.5,
    }, index=index)
    evidence.loc[index[1], "gap_load"] = 2.0
    evidence.loc[index[1], "range_pressure"] = -0.5
    evidence.loc[index[4], :] = np.nan
    result = GeneratedStrategy.generate_signals(None, data, evidence, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
