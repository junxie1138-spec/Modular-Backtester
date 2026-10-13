from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapPropagationParams:
    baseline_window: int = 40
    incidence_window: int = 6
    gap_threshold: float = 0.75
    min_incidence: float = 0.5
    min_growth: float = 0.16
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapPropagationParams]):
    strategy_id = "gen_local_1791254909"

    @classmethod
    def params_type(cls):
        return GapPropagationParams

    def warmup_bars(self, params: GapPropagationParams) -> int:
        return int(params.baseline_window + 2 * params.incidence_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapPropagationParams) -> pd.DataFrame:
        if params.baseline_window < 2 or params.incidence_window < 2 or params.hold_bars < 1:
            raise ValueError("Windows must be at least two and hold_bars positive.")
        if not np.isfinite(params.gap_threshold) or params.gap_threshold <= 0:
            raise ValueError("gap_threshold must be finite and positive.")
        if not 0 < params.min_incidence < 1 or not 0 < params.min_growth <= 1:
            raise ValueError("Invalid incidence or growth threshold.")
        out = pd.DataFrame(index=data.index)
        prior_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / prior_close - 1).replace([np.inf, -np.inf], np.nan)
        out["gap_scale"] = out["gap"].abs().rolling(params.baseline_window).mean().shift(1)
        out["normalized_gap"] = (out["gap"] / out["gap_scale"].where(out["gap_scale"] > 0)).replace([np.inf, -np.inf], np.nan)
        valid = out["normalized_gap"].notna()
        out["up_event"] = (out["normalized_gap"] >= params.gap_threshold).astype(float).where(valid)
        out["down_event"] = (out["normalized_gap"] <= -params.gap_threshold).astype(float).where(valid)
        for direction in ("up", "down"):
            out[direction + "_incidence"] = out[direction + "_event"].rolling(params.incidence_window).mean()
            incidence = out[direction + "_incidence"]
            out[direction + "_previous"] = incidence.shift(params.incidence_window)
            out[direction + "_growth"] = incidence - out[direction + "_previous"]
            growth = out[direction + "_growth"]
            magnitude = out["normalized_gap"].abs().clip(upper=3.0)
            out[direction + "_strength"] = (incidence * growth.clip(lower=0) * (1 - incidence) * magnitude / 0.25).clip(0, 1).fillna(0.0)
            out[direction + "_eligible"] = ((out[direction + "_event"] == 1) & (incidence >= params.min_incidence) & (incidence < 1) & (growth >= params.min_growth)).fillna(False)
        out["entry_direction"] = np.select([out["up_eligible"], out["down_eligible"]], [1, -1], default=0).astype(int)
        out["conviction"] = np.where(out["entry_direction"] == 1, out["up_strength"], np.where(out["entry_direction"] == -1, out["down_strength"], 0.0))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapPropagationParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        conviction = indicators["conviction"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        side = 0
        exit_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if side != 0 and i == exit_bar:
                side = 0
            elif side == 0 and entries[i] != 0:
                side = int(entries[i])
                exit_bar = i + params.hold_bars
                entry_size = 0.2 + 0.8 * conviction[i]
            signals[i] = side
            if side != 0:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapPropagationParams(hold_bars=18)
    index = pd.date_range("2020-01-01", periods=45, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry_direction": 1, "conviction": 0.5}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].iloc[0] == 0
    assert (result["signal"].iloc[1:19] == 1).all()
    assert result["signal"].iloc[19] == 0
    assert (result["signal"].iloc[20:38] == 1).all()
    assert result["signal"].iloc[38] == 0
    assert np.allclose(result["size"].iloc[1:19], 0.6)
    indicators["entry_direction"] = -1
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert (result["signal"].iloc[1:19] == -1).all()
    assert result["signal"].iloc[19] == 0


if __name__ == "__main__":
    _self_check()
