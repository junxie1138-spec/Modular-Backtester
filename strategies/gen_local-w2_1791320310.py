from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTranslationParams:
    box_window: int = 18
    translation_threshold: float = 0.20
    gap_threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[RangeTranslationParams]):
    strategy_id = "gen_local-w2_1791320310"

    @classmethod
    def params_type(cls) -> type[RangeTranslationParams]:
        return RangeTranslationParams

    def warmup_bars(self, params: RangeTranslationParams) -> int:
        return int(max(2 * params.box_window, params.box_window + 1))

    def indicators(self, data: pd.DataFrame, params: RangeTranslationParams) -> pd.DataFrame:
        if params.box_window < 2:
            raise ValueError("box_window must be at least 2")
        if not np.isfinite(params.translation_threshold) or params.translation_threshold <= 0:
            raise ValueError("translation_threshold must be positive and finite")
        if not np.isfinite(params.gap_threshold) or not 0 < params.gap_threshold <= 1:
            raise ValueError("gap_threshold must be in (0, 1]")
        n = params.box_window
        out = pd.DataFrame(index=data.index)
        out["box_high"] = data["high"].rolling(n, min_periods=n).max()
        out["box_low"] = data["low"].rolling(n, min_periods=n).min()
        out["prior_high"] = out["box_high"].shift(n)
        out["prior_low"] = out["box_low"].shift(n)
        out["midpoint"] = (out["box_high"] + out["box_low"]) / 2.0
        out["prior_midpoint"] = (out["prior_high"] + out["prior_low"]) / 2.0
        out["width"] = out["box_high"] - out["box_low"]
        out["prior_width"] = out["prior_high"] - out["prior_low"]
        out["displacement"] = out["midpoint"] - out["prior_midpoint"]
        out["deformation"] = (out["width"] - out["prior_width"]).abs() / 2.0
        out["scale"] = (out["width"] + out["prior_width"]) / 2.0
        out["persistent_translation"] = (out["displacement"].abs() - out["deformation"]).clip(lower=0.0)
        out["range_translation"] = np.sign(out["displacement"]) * out["persistent_translation"] / out["scale"].where(out["scale"] > 0)
        out["overnight_gap"] = data["open"] - data["close"].shift(1)
        out["net_gap"] = out["overnight_gap"].rolling(n, min_periods=n).sum()
        out["absolute_gap"] = out["overnight_gap"].abs().rolling(n, min_periods=n).sum()
        out["overnight_acceptance"] = out["net_gap"] / out["absolute_gap"].where(out["absolute_gap"] > 0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeTranslationParams) -> SignalFrame:
        translation = indicators["range_translation"].fillna(0.0)
        acceptance = indicators["overnight_acceptance"].fillna(0.0)
        long_entry = (translation >= params.translation_threshold) & (acceptance >= params.gap_threshold)
        short_entry = (translation <= -params.translation_threshold) & (acceptance <= -params.gap_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = RangeTranslationParams()
    strategy = object.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"range_translation": [0.4, 0.0, -0.4, -0.4, 0.4], "overnight_acceptance": [0.4, 0.0, -0.4, -0.4, 0.4]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 2 * params.box_window
