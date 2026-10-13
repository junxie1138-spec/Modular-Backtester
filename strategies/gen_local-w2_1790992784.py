from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEnergyParams:
    energy_window: int = 8
    percentile_window: int = 126
    energy_percentile: float = 0.80
    flow_percentile: float = 0.65
    profit_target: float = 0.025
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GapEnergyParams]):
    strategy_id = "gen_local-w2_1790992784"

    @classmethod
    def params_type(cls):
        return GapEnergyParams

    def warmup_bars(self, params: GapEnergyParams) -> int:
        return int(params.percentile_window + params.energy_window + 9)

    def indicators(self, data: pd.DataFrame, params: GapEnergyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["day_move"] = ((data["close"] - data["open"]) / previous_close).replace([np.inf, -np.inf], np.nan)
        out["close_move"] = out["gap"] + out["day_move"]
        for half in (1, 2, 4):
            out[f"gap_sum_{half}"] = out["gap"].rolling(half, min_periods=half).sum()
            out[f"haar_{2 * half}"] = (
                out[f"gap_sum_{half}"] - out[f"gap_sum_{half}"].shift(half)
            ) / np.sqrt(2.0 * half)
            out[f"energy_{2 * half}"] = out[f"haar_{2 * half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["burst_ratio"] = out["energy_2"] / (
            out["energy_4"] + out["energy_8"]
        ).replace(0.0, np.nan)
        out["burst_threshold"] = out["burst_ratio"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.energy_percentile)
        out["gap_flow"] = out["gap"].rolling(4, min_periods=4).sum()
        out["flow_magnitude"] = out["gap_flow"].abs()
        out["flow_threshold"] = out["flow_magnitude"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.flow_percentile)
        out["direction"] = np.sign(out["gap_flow"])
        out["aligned_gap"] = out["direction"] * out["gap"]
        out["aligned_day"] = out["direction"] * out["day_move"]
        out["retained_gap"] = out["direction"] * out["close_move"]
        ready = (
            (out["burst_ratio"] > out["burst_threshold"])
            & (out["flow_magnitude"] > out["flow_threshold"])
            & (out["aligned_gap"] > 0.0)
            & (out["aligned_day"] < 0.0)
            & (out["retained_gap"] >= 0.5 * out["aligned_gap"])
        )
        out["entry_signal"] = out["direction"].where(ready, 0).fillna(0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapEnergyParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        candidates = indicators["entry_signal"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        pending_entry = False
        for i in range(len(data)):
            if pending_entry:
                # The prior close's decision fills at this bar's open.
                entry_price = opens[i]
                entry_bar = i
                pending_entry = False
            if position:
                age = i - entry_bar + 1
                valid_price = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_price else np.nan
                if (
                    not valid_price
                    or (np.isfinite(gain) and gain >= params.profit_target)
                    or age >= params.max_hold_bars
                ):
                    position = 0
                    entry_price = np.nan
                    # Keep the exit decision flat; do not re-enter on this close.
                decisions[i] = position
                continue
            if candidates[i] != 0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(candidates[i])
                pending_entry = True
            decisions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
