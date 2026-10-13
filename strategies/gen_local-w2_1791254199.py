from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapInventoryParams:
    gap_budget: float = 0.012
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[GapInventoryParams]):
    strategy_id = "gen_local-w2_1791254199"

    @classmethod
    def params_type(cls) -> type[GapInventoryParams]:
        return GapInventoryParams

    def warmup_bars(self, params: GapInventoryParams) -> int:
        return 50

    def indicators(self, data: pd.DataFrame, params: GapInventoryParams) -> pd.DataFrame:
        if not np.isfinite(params.gap_budget) or params.gap_budget <= 0:
            raise ValueError("gap_budget must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["gap_amount"] = data["open"] - out["previous_close"]
        out["gap_return"] = out["gap_amount"] / out["previous_close"].where(out["previous_close"] > 0)
        out["retained_fraction"] = (
            (data["low"] - out["previous_close"])
            / out["gap_amount"].where(out["gap_amount"] > 0)
        ).clip(0.0, 1.0).fillna(0.0)
        out["retained_gap"] = out["gap_return"].clip(lower=0.0) * out["retained_fraction"]
        out["gap_inventory"] = out["retained_gap"].rolling(12, min_periods=12).sum()
        out["older_inventory"] = out["gap_inventory"].shift(6)
        out["defended_gap"] = ((out["gap_amount"] > 0) & (out["retained_fraction"] > 0)).astype(int)
        out["defended_count"] = out["defended_gap"].rolling(12, min_periods=12).sum()
        out["intraday_return"] = data["close"] / data["open"].where(data["open"] > 0) - 1.0
        out["intraday_drift"] = out["intraday_return"].rolling(12, min_periods=12).sum()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["trend"] = data["close"].rolling(50, min_periods=50).mean()
        out["entry_ready"] = (
            (out["gap_inventory"] > params.gap_budget)
            & (out["gap_inventory"] > out["older_inventory"])
            & (out["defended_count"] >= 3)
            & (out["intraday_drift"] < 0)
            & (out["gap_inventory"] + out["intraday_drift"] > 0)
            & (data["close"] > out["trend"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapInventoryParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        armed = False
        entry_price = 0.0
        peak_close = 0.0
        stop = 0.0
        entry_bar = 0

        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    armed = False
                    entry_price = opens[i]
                    peak_close = entry_price
                    stop = max(0.0, entry_price - 2.0 * atr[i - 1])
                    entry_bar = i

            if active:
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                peak_close = max(peak_close, closes[i])
                if closes[i] <= stop or i - entry_bar + 1 >= 18:
                    active = False
                    continue
                if closes[i] >= entry_price * 1.02:
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak_close - params.trail_atr * atr[i])
                    if closes[i] <= stop:
                        active = False
                        continue
                signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0:
                signals[i] = 1
                pending_entry = True

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
