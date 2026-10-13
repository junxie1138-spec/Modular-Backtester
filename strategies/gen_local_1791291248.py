from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    regime_window: int = 3
    rank_window: int = 60
    atr_window: int = 14
    percentile: float = 0.85
    stop_atr: float = 1.2
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791291248"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.rank_window + 2 * params.regime_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.regime_window, params.rank_window, params.atr_window) < 1:
            raise ValueError("Lookbacks must be positive")
        if not 0.0 < params.percentile <= 1.0 or params.stop_atr <= 0.0:
            raise ValueError("Invalid percentile or stop multiplier")
        if params.hold_days not in (1, 2):
            raise ValueError("hold_days must be one or two")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(params.atr_window).mean()
        floor = data["low"].rolling(params.regime_window).min()
        out["recent_floor"] = floor
        out["previous_floor"] = floor.shift(params.regime_window)
        out["floor_jump"] = (out["recent_floor"] - out["previous_floor"]) / out["atr"].replace(0.0, np.nan)
        close_change = data["close"].diff()
        downside_volume = data["volume"].where(close_change < 0.0, 0.0).where(close_change.notna())
        out["downside_volume"] = downside_volume
        out["total_volume"] = data["volume"].rolling(params.regime_window).sum()
        out["downside_volume_sum"] = downside_volume.rolling(params.regime_window).sum()
        out["downside_share"] = out["downside_volume_sum"] / out["total_volume"].replace(0.0, np.nan)
        out["previous_downside_share"] = out["downside_share"].shift(params.regime_window)
        out["selling_withdrawal"] = out["previous_downside_share"] - out["downside_share"]
        out["floor_percentile"] = out["floor_jump"].rolling(params.rank_window).rank(pct=True)
        out["withdrawal_percentile"] = out["selling_withdrawal"].rolling(params.rank_window).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        agreement = (
            (indicators["floor_jump"] > 0.0)
            & (indicators["floor_percentile"] >= params.percentile)
            & (indicators["selling_withdrawal"] > 0.0)
            & (indicators["withdrawal_percentile"] >= params.percentile)
        ).fillna(False)
        entry_event = agreement & ~agreement.shift(1, fill_value=False)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = entry_event.to_numpy(dtype=bool)
        desired = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        held_days = 0
        stop_price = np.nan
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0.0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0.0:
                    active = False
                else:
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]
                    held_days = 0
            if active:
                held_days += 1
                if not np.isfinite(closes[i]) or closes[i] < stop_price or held_days >= params.hold_days:
                    active = False
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0.0 and np.isfinite(closes[i]) and closes[i] > 0.0:
                active = True
                pending_entry = True
            desired[i] = int(active)
        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        assert df["signal"].isin([0, 1]).all() and df["size"].gt(0.0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
