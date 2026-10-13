from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    energy_window: int = 3
    percentile_window: int = 120
    entry_percentile: float = 0.80
    profit_target: float = 0.015
    max_hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local-w2_1790986458"

    @classmethod
    def params_type(cls):
        return GapRepairParams

    def warmup_bars(self, params: GapRepairParams) -> int:
        return 8 + params.energy_window + params.percentile_window + 1

    def indicators(self, data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["gap"] = np.log(opening / close.shift(1))
        out["intraday"] = np.log(close / opening)
        out["prior_high"] = data["high"].shift(1)
        out["prior_low"] = data["low"].shift(1)
        out["abs_gap"] = out["gap"].abs()
        for width in (2, 4, 8):
            half = width // 2
            recent = out["gap"].rolling(half, min_periods=half).sum()
            out[f"haar_{width}"] = (recent - recent.shift(half)) / np.sqrt(width)
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(
            axis=1, min_count=3
        )
        out["fine_share"] = out["energy_2"] / out["total_energy"].where(
            out["total_energy"] > 0
        )
        for column in ("abs_gap", "fine_share"):
            out[f"{column}_threshold"] = out[column].shift(1).rolling(
                params.percentile_window, min_periods=params.percentile_window
            ).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRepairParams,
    ) -> SignalFrame:
        ready = indicators[[
            "gap", "intraday", "prior_high", "prior_low", "abs_gap",
            "fine_share", "abs_gap_threshold", "fine_share_threshold"
        ]].notna().all(axis=1)
        repaired = data["close"].between(
            indicators["prior_low"], indicators["prior_high"]
        )
        shock = (
            ready
            & (indicators["abs_gap"] > indicators["abs_gap_threshold"])
            & (indicators["fine_share"] > indicators["fine_share_threshold"])
            & repaired
        )
        candidates = pd.Series(0, index=data.index, dtype=int)
        candidates.loc[
            shock & (indicators["gap"] > 0) & (indicators["intraday"] < 0)
        ] = -1
        candidates.loc[
            shock & (indicators["gap"] < 0) & (indicators["intraday"] > 0)
        ] = 1
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = candidates.to_numpy(dtype=int)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        # Signals decided at this close execute at the following open.
        for i in range(len(data)):
            if direction != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                age = i - entry_bar + 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = (
                    direction * (closes[i] / entry_price - 1.0)
                    if valid_entry and valid_close else -np.inf
                )
                if (
                    not valid_entry
                    or gain >= params.profit_target
                    or age >= params.max_hold_bars
                ):
                    direction = 0
                    entry_price = np.nan
                    raw[i] = 0
                    continue
            elif entries[i] != 0:
                direction = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = direction
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
