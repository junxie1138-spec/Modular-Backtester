from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockCoolingParams:
    shock_multiple: float = 2.5
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[ShockCoolingParams]):
    strategy_id = "gen_local_1790993540"

    @classmethod
    def params_type(cls) -> type[ShockCoolingParams]:
        return ShockCoolingParams

    def warmup_bars(self, params: ShockCoolingParams) -> int:
        return 20

    def indicators(self, data: pd.DataFrame, params: ShockCoolingParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["close_change"] = data["close"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_atr"] = out["atr"].shift(1)
        denominator = out["previous_atr"].where(out["previous_atr"] > 0)
        out["range_ratio"] = out["true_range"] / denominator
        out["shock"] = (
            (out["range_ratio"] >= params.shock_multiple)
            & (out["close_change"].abs() >= denominator)
        ).fillna(False)
        out["shock_direction"] = np.sign(out["close_change"]).where(out["shock"]).ffill()
        out["anchor"] = out["previous_close"].where(out["shock"]).ffill()
        out["shock_atr"] = out["previous_atr"].where(out["shock"]).ffill()
        out["bar_number"] = np.arange(len(data), dtype=float)
        out["last_shock_bar"] = out["bar_number"].where(out["shock"]).ffill()
        out["shock_age"] = out["bar_number"] - out["last_shock_bar"]
        out["residual"] = (
            out["shock_direction"] * (data["close"] - out["anchor"])
            / out["shock_atr"].where(out["shock_atr"] > 0)
        )
        out["cooled"] = (out["true_range"] <= out["previous_atr"]).fillna(False)
        out["entry_ready"] = (
            (out["shock_age"] == 3)
            & out["cooled"]
            & (out["residual"] >= 0.5)
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockCoolingParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        direction = indicators["shock_direction"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        stop_distance = np.nan

        for i in range(len(data)):
            if position != 0:
                # Yesterday's decision fills at today's open. Freeze its ATR.
                if i == entry_bar:
                    entry_price = opens[i]
                if (
                    not np.isfinite(entry_price)
                    or entry_price <= 0
                    or not np.isfinite(closes[i])
                    or closes[i] <= 0
                ):
                    position = 0
                    continue
                # Long: close <= entry - k*ATR; short: symmetric upper stop.
                stopped = position * (closes[i] - entry_price) <= -stop_distance
                expired = i - entry_bar >= 7
                if stopped or expired:
                    position = 0
                signals[i] = position
                continue

            if (
                ready[i]
                and np.isfinite(direction[i])
                and direction[i] != 0
                and np.isfinite(atr[i])
                and atr[i] > 0
                and np.isfinite(closes[i])
                and closes[i] > 0
            ):
                position = -int(direction[i])
                entry_bar = i + 1
                entry_price = np.nan
                stop_distance = params.stop_atr * atr[i]
                signals[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
