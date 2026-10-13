from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMigrationParams:
    prior_rank_min: float = 0.65
    current_rank_max: float = 0.40
    activation_pct: float = 0.012
    trail_atr: float = 1.6
    max_hold: int = 5


class GeneratedStrategy(BaseStrategy[GapMigrationParams]):
    strategy_id = "gen_local_1791295835"

    @classmethod
    def params_type(cls) -> type[GapMigrationParams]:
        return GapMigrationParams

    def warmup_bars(self, params: GapMigrationParams) -> int:
        return 66

    def indicators(self, data: pd.DataFrame, params: GapMigrationParams) -> pd.DataFrame:
        previous_close = data["close"].shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        gap_share = ((data["open"] - previous_close).abs()
                     .div(true_range.replace(0, np.nan)).fillna(0).clip(0, 1))
        rank = gap_share.rolling(3, min_periods=3).mean().rolling(63, min_periods=63).rank(pct=True)
        return pd.DataFrame({
            "gap_rank": rank.fillna(0),
            "atr": true_range.rolling(14, min_periods=14).mean().fillna(0),
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapMigrationParams,
    ) -> SignalFrame:
        rank = indicators["gap_rank"]
        entry_setup = ((rank.shift(1) >= params.prior_rank_min)
                       & (rank <= params.current_rank_max)
                       & (data["close"] > data["open"]))
        signals = np.zeros(len(data), dtype=int)
        pending_entry = False
        in_position = False
        entry = stop = peak = 0.0
        days = 0

        for i in range(len(data)):
            if pending_entry:
                entry = float(data["open"].iat[i])
                peak = entry
                stop = entry - params.trail_atr * float(indicators["atr"].iat[i - 1])
                days = 0
                in_position = True
                pending_entry = False

            if in_position:
                days += 1
                if float(data["low"].iat[i]) <= stop or days >= params.max_hold:
                    in_position = False
                else:
                    peak = max(peak, float(data["high"].iat[i]))
                    if peak >= entry * (1 + params.activation_pct):
                        stop = max(stop, entry, peak - params.trail_atr * float(indicators["atr"].iat[i]))
                    signals[i] = 1
            elif bool(entry_setup.iat[i]):
                signals[i] = 1
                pending_entry = True

        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
