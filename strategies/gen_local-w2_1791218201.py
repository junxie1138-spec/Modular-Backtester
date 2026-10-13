from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapExhaustionParams:
    peak_window: int = 40
    min_drawdown: float = 0.03
    max_gap_ratio: float = 0.65
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapExhaustionParams]):
    strategy_id = "gen_local-w2_1791218201"

    @classmethod
    def params_type(cls) -> type[GapExhaustionParams]:
        return GapExhaustionParams

    @staticmethod
    def warmup_bars(params: GapExhaustionParams) -> int:
        return int(max(params.peak_window + 1, 3))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapExhaustionParams) -> pd.DataFrame:
        if params.peak_window < 2 or not 3 <= params.hold_bars <= 5:
            raise ValueError("peak_window must be >= 2 and hold_bars must be 3 to 5")
        if not 0 < params.min_drawdown < 1 or not 0 < params.max_gap_ratio < 1:
            raise ValueError("drawdown and gap ratio thresholds must be between 0 and 1")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        out["prior_peak"] = data["high"].rolling(
            params.peak_window, min_periods=params.peak_window
        ).max().shift(1).where(lambda x: x > 0)
        out["drawdown"] = (1.0 - data["close"] / out["prior_peak"]).clip(lower=0)
        out["previous_drawdown"] = out["drawdown"].shift(1)
        out["gap"] = opening / previous_close - 1.0
        out["previous_gap"] = out["gap"].shift(1)
        out["intraday_return"] = data["close"] / opening - 1.0
        out["gap_ratio"] = out["gap"].abs() / out["previous_gap"].abs().replace(0, np.nan)
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out.notna().all(axis=1) & (data["close"] > 0)
        out["entry"] = (
            valid
            & (out["drawdown"] >= params.min_drawdown)
            & (out["drawdown"] > out["previous_drawdown"])
            & (out["gap"] < 0)
            & (out["previous_gap"] < 0)
            & (out["gap_ratio"] <= params.max_gap_ratio)
            & (out["intraday_return"] <= 0)
        ).fillna(False)
        out["strength"] = (
            (out["drawdown"] / (2.0 * params.min_drawdown)).clip(0, 1)
            * (1.0 - out["gap_ratio"]).clip(0, 1)
        ).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapExhaustionParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be 3 to 5")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        strength = indicators["strength"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        exit_at = -1
        entry_size = 0.25
        for i in range(len(data)):
            if i == exit_at:
                continue
            if i > exit_at and entries[i]:
                exit_at = i + params.hold_bars
                entry_size = 0.25 + 0.70 * strength[i]
            if i < exit_at:
                signals[i] = 1
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"entry": True, "strength": np.arange(12, dtype=float) / 10.0},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, indicators, None, GapExhaustionParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert np.allclose(result["size"].iloc[1:5], 0.25)
    assert np.allclose(result["size"].iloc[6:10], 0.60)
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
