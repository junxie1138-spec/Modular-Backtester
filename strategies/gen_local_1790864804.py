from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapQueueParams:
    history_window: int = 5
    scale_window: int = 20
    min_down_gaps: int = 3
    debt_threshold: float = 1.5
    attenuation_ratio: float = 0.6
    absorption_floor: float = 0.5
    hold_bars: int = 4
    min_size: float = 0.2
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapQueueParams]):
    strategy_id = "gen_local_1790864804"

    @classmethod
    def params_type(cls) -> type[GapQueueParams]:
        return GapQueueParams

    def warmup_bars(self, params: GapQueueParams) -> int:
        return int(max(params.history_window, params.scale_window) + 2)

    def indicators(self, data: pd.DataFrame, params: GapQueueParams) -> pd.DataFrame:
        if params.history_window < 1 or params.scale_window < 1:
            raise ValueError("Lookbacks must be positive")
        if not 1 <= params.min_down_gaps <= params.history_window:
            raise ValueError("Invalid minimum gap count")
        if params.debt_threshold <= 0 or not 0 < params.attenuation_ratio < 1:
            raise ValueError("Invalid pressure or attenuation threshold")
        if not 0 <= params.absorption_floor <= 1:
            raise ValueError("Invalid absorption floor")
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        out = pd.DataFrame(index=data.index)
        out["gap"] = data["open"] / previous_close - 1.0
        out["intraday"] = (data["close"] - data["open"]) / previous_close
        out["gap_loss"] = (-out["gap"]).clip(lower=0.0)
        out["gap_scale"] = out["gap"].abs().rolling(params.scale_window).mean().shift(1)
        out["prior_debt"] = out["gap_loss"].rolling(params.history_window).sum().shift(1)
        down_gap = out["gap"].lt(0).astype(float).where(out["gap"].notna())
        out["prior_count"] = down_gap.rolling(params.history_window).sum().shift(1)
        out["prior_intraday"] = out["intraday"].rolling(params.history_window).sum().shift(1)
        out["mean_down_gap"] = out["prior_debt"] / out["prior_count"].replace(0.0, np.nan)
        out["debt_strength"] = out["prior_debt"] / out["gap_scale"].replace(0.0, np.nan)
        out["absorption"] = out["prior_intraday"] / out["prior_debt"].replace(0.0, np.nan)
        out["shock_ratio"] = out["gap_loss"] / out["mean_down_gap"].replace(0.0, np.nan)
        out["recovery_fraction"] = out["intraday"] / out["gap_loss"].replace(0.0, np.nan)
        day_range = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["close_location"] = (data["close"] - data["low"]) / day_range
        out["conviction"] = (
            out["debt_strength"]
            * (1.0 - out["shock_ratio"]).clip(0.0, 1.0)
            * out["absorption"].clip(0.0, 1.0)
        )
        out["entry_ready"] = (
            out["gap"].lt(0.0)
            & out["gap_scale"].gt(0.0)
            & out["prior_count"].ge(params.min_down_gaps)
            & out["debt_strength"].ge(params.debt_threshold)
            & out["absorption"].ge(params.absorption_floor)
            & out["shock_ratio"].le(params.attenuation_ratio)
            & out["recovery_fraction"].ge(0.5)
            & out["close_location"].ge(0.65)
        )
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapQueueParams,
    ) -> SignalFrame:
        if params.hold_bars < 1 or not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Invalid holding period or position sizes")
        ready = indicators["entry_ready"].fillna(False).astype(bool).to_numpy()
        conviction = indicators["conviction"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        entry_sizes = (
            params.min_size
            + (params.max_size - params.min_size) * conviction / (1.0 + conviction)
        ).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        held_size = params.min_size
        for i in range(len(data)):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = held_size
            elif ready[i]:
                exit_bar = i + params.hold_bars
                held_size = float(entry_sizes[i])
                signals[i] = 1
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry_ready": True, "conviction": 1.0}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, GapQueueParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    assert result.data["size"].gt(0.0).all()
