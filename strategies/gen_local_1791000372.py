from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    min_loss_streak: int = 3
    recovery_fraction: float = 0.30
    stop_atr: float = 1.5
    atr_period: int = 14
    peak_period: int = 20
    min_drawdown_atr: float = 0.75
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791000372"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.atr_period + 1, params.peak_period + 1, params.min_loss_streak + 2))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["close_change"] = close.diff()
        out["down_day"] = out["close_change"].lt(0)
        out["streak_group"] = (~out["down_day"]).cumsum()
        out["loss_streak"] = out["down_day"].astype(int).groupby(out["streak_group"]).cumsum()
        out["previous_loss_streak"] = out["loss_streak"].shift(1)
        out["streak_anchor"] = close.where(~out["down_day"]).ffill()
        out["previous_streak_anchor"] = out["streak_anchor"].shift(1)
        out["streak_damage"] = out["previous_streak_anchor"] - out["previous_close"]
        out["reclaimed_fraction"] = out["close_change"] / out["streak_damage"].where(out["streak_damage"].gt(0))
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["recent_peak"] = close.rolling(params.peak_period, min_periods=params.peak_period).max()
        out["previous_peak"] = out["recent_peak"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["previous_drawdown_atr"] = (out["previous_peak"] - out["previous_close"]) / out["previous_atr"].where(out["previous_atr"].gt(0))
        out["entry_ready"] = (
            out["previous_loss_streak"].ge(params.min_loss_streak)
            & out["close_change"].gt(0)
            & out["reclaimed_fraction"].ge(params.recovery_fraction)
            & out["previous_drawdown_atr"].ge(params.min_drawdown_atr)
            & close.gt(out["ma200"])
            & out["atr"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        count = len(data)
        desired = np.zeros(count, dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_ready"].to_numpy(dtype=bool)
        holding_limit = max(1, min(2, int(params.hold_bars)))
        active = False
        entry_index = -1
        frozen_atr = np.nan
        stop_price = np.nan

        for i in range(count):
            if active:
                if i == entry_index:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    stop_price = opens[i] - params.stop_atr * frozen_atr
                held_sessions = i - entry_index + 1
                stop_hit = np.isfinite(closes[i]) and closes[i] < stop_price
                if stop_hit or held_sessions >= holding_limit:
                    active = False
                else:
                    desired[i] = 1
            elif entries[i] and np.isfinite(atrs[i]) and atrs[i] > 0:
                active = True
                entry_index = i + 1
                frozen_atr = atrs[i]
                desired[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = desired
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
