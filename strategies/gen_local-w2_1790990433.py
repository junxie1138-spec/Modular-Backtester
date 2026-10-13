from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ChangeParams:
    baseline_bars: int = 30
    reaction_bars: int = 3
    atr_bars: int = 14
    change_threshold: float = 1.8
    volume_threshold: float = 1.1
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ChangeParams]):
    strategy_id = "gen_local-w2_1790990433"

    @classmethod
    def params_type(cls) -> type[ChangeParams]:
        return ChangeParams

    def warmup_bars(self, params: ChangeParams) -> int:
        n = max(2, int(params.baseline_bars))
        r = max(2, int(params.reaction_bars))
        a = max(1, int(params.atr_bars))
        return int(max(n, a) + r + n + 2)

    def indicators(self, data: pd.DataFrame, params: ChangeParams) -> pd.DataFrame:
        n = max(2, int(params.baseline_bars))
        r = max(2, int(params.reaction_bars))
        a = max(1, int(params.atr_bars))
        close = data["close"].astype(float)
        volume = data["volume"].astype(float).clip(lower=0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(a, min_periods=a).mean()
        reference_atr = atr.shift(1).replace(0.0, np.nan)
        reference_volume = volume.shift(1).rolling(n, min_periods=n).mean()
        participation = volume.div(reference_volume.replace(0.0, np.nan))
        impact = close.diff().div(reference_atr) * participation.clip(upper=3.0)
        old_impact = impact.shift(r)
        old_mean = old_impact.rolling(n, min_periods=n).mean()
        old_std = old_impact.rolling(n, min_periods=n).std(ddof=0)
        recent_mean = impact.rolling(r, min_periods=r).mean()
        shift_score = (recent_mean - old_mean).div(old_std.clip(lower=0.15))
        recent_participation = participation.rolling(r, min_periods=r).mean()
        net_move = close.diff(r).div(reference_atr)
        return pd.DataFrame({
            "atr": atr,
            "reference_atr": reference_atr,
            "reference_volume": reference_volume,
            "participation": participation,
            "impact": impact,
            "old_mean": old_mean,
            "old_std": old_std,
            "recent_mean": recent_mean,
            "shift_score": shift_score,
            "recent_participation": recent_participation,
            "net_move": net_move,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ChangeParams,
    ) -> SignalFrame:
        ready = indicators.notna().all(axis=1) & (indicators["atr"] > 0.0)
        active_volume = (
            (indicators["recent_participation"] >= params.volume_threshold)
            & (indicators["participation"] >= 1.0)
        )
        bullish = (
            ready & active_volume
            & (indicators["shift_score"] >= params.change_threshold)
            & (indicators["recent_mean"] > 0.0)
            & (indicators["net_move"] > 0.0)
        )
        bearish = (
            ready & active_volume
            & (indicators["shift_score"] <= -params.change_threshold)
            & (indicators["recent_mean"] < 0.0)
            & (indicators["net_move"] < 0.0)
        )
        confirmed_long = bullish & bullish.shift(1, fill_value=False)
        confirmed_short = bearish & bearish.shift(1, fill_value=False)
        # One entry opportunity per newly confirmed regime.
        long_event = confirmed_long & ~confirmed_long.shift(1, fill_value=False)
        short_event = confirmed_short & ~confirmed_short.shift(1, fill_value=False)
        entries = np.where(long_event, 1, np.where(short_event, -1, 0))
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        stop_price = np.nan
        frozen_atr = np.nan
        held = 0
        holding_limit = max(1, int(params.hold_bars))
        stop_multiple = max(0.01, float(params.stop_atr))

        for i in range(len(data)):
            if position != 0:
                # Yesterday's entry decision fills at today's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                    stop_price = entry_price - position * stop_multiple * frozen_atr
                held += 1
                valid = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(closes[i]) and closes[i] > 0.0
                )
                stopped = valid and (
                    (position == 1 and closes[i] < stop_price)
                    or (position == -1 and closes[i] > stop_price)
                )
                if not valid or stopped or held >= holding_limit:
                    position = 0
                    entry_price = np.nan
                    stop_price = np.nan
                    held = 0
                signals[i] = position
                # Exit decisions cannot also initiate a replacement position.
                continue

            if entries[i] != 0 and np.isfinite(atrs[i]) and atrs[i] > 0.0:
                position = int(entries[i])
                frozen_atr = atrs[i]
                entry_price = np.nan
                held = 0
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
