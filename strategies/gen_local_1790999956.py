from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShelfParams:
    envelope_bars: int = 20
    shelf_bars: int = 5
    volume_bars: int = 10
    atr_bars: int = 14
    floor_position: float = 0.55
    close_position: float = 0.70
    up_volume_share: float = 0.62
    min_volume_ratio: float = 0.95
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[ShelfParams]):
    strategy_id = "gen_local_1790999956"

    @classmethod
    def params_type(cls) -> type[ShelfParams]:
        return ShelfParams

    @staticmethod
    def warmup_bars(params: ShelfParams) -> int:
        return int(max(
            params.envelope_bars + params.shelf_bars,
            params.volume_bars + 1,
            params.atr_bars + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: ShelfParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float).clip(lower=0)
        out["old_high"] = data["high"].rolling(params.envelope_bars).max().shift(params.shelf_bars)
        out["old_low"] = data["low"].rolling(params.envelope_bars).min().shift(params.shelf_bars)
        out["old_width"] = (out["old_high"] - out["old_low"]).replace(0.0, np.nan)
        out["shelf_floor"] = data["low"].rolling(params.shelf_bars).min()
        out["shelf_ceiling"] = data["high"].rolling(params.shelf_bars).max()
        out["floor_location"] = (out["shelf_floor"] - out["old_low"]) / out["old_width"]
        out["close_location"] = (close - out["old_low"]) / out["old_width"]
        out["close_change"] = close.diff()
        out["up_volume"] = volume.where(out["close_change"] > 0, 0.0).where(out["close_change"].notna())
        out["down_volume"] = volume.where(out["close_change"] < 0, 0.0).where(out["close_change"].notna())
        out["up_volume_sum"] = out["up_volume"].rolling(params.volume_bars).sum()
        out["down_volume_sum"] = out["down_volume"].rolling(params.volume_bars).sum()
        out["directional_volume"] = out["up_volume_sum"] + out["down_volume_sum"]
        out["up_share"] = out["up_volume_sum"] / out["directional_volume"].replace(0.0, np.nan)
        out["recent_volume"] = volume.rolling(params.volume_bars).mean()
        out["old_volume"] = volume.rolling(params.envelope_bars).mean().shift(params.shelf_bars)
        out["volume_ratio"] = out["recent_volume"] / out["old_volume"].replace(0.0, np.nan)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["position_agrees"] = (
            (out["floor_location"] >= params.floor_position)
            & (out["close_location"] >= params.close_position)
            & (out["close_location"] <= 1.0)
            & (out["shelf_ceiling"] <= out["old_high"])
        ).fillna(False)
        out["volume_agrees"] = (
            (out["up_share"] >= params.up_volume_share)
            & (out["volume_ratio"] >= params.min_volume_ratio)
        ).fillna(False)
        out["entry"] = (
            out["position_agrees"] & out["volume_agrees"]
            & out["atr"].gt(0) & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShelfParams,
    ) -> SignalFrame:
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry_price = 0.0
        entry_bar = 0
        peak = 0.0
        stop = 0.0

        for i in range(n):
            # A decision from the previous close enters at today's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and i > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    armed = False
                    entry_price = opens[i]
                    entry_bar = i
                    peak = entry_price
                    stop = entry_price - params.trail_atr * atr[i - 1]

            if active:
                # Stops are close-observed exit decisions, filled next bar.
                # Test the existing stop before raising it with today's high.
                stop_hit = np.isfinite(lows[i]) and lows[i] <= stop
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if stop_hit or expired:
                    active = False
                    continue
                if np.isfinite(highs[i]):
                    peak = max(peak, highs[i])
                if peak >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atr[i]) and atr[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atr[i])
                decisions[i] = 1
            elif entries[i]:
                pending = True
                decisions[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")
