from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapFailureParams:
    tension_threshold: float = 0.8
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[GapFailureParams]):
    strategy_id = "gen_local_1790986875"

    @classmethod
    def params_type(cls) -> type[GapFailureParams]:
        return GapFailureParams

    def warmup_bars(self, params: GapFailureParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: GapFailureParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        out["return"] = (data["close"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["gap"] = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        out["return_volatility"] = out["return"].rolling(20, min_periods=20).std(ddof=0).shift(1)
        out["return_sum"] = out["return"].rolling(3, min_periods=3).sum()
        out["gap_sum"] = out["gap"].rolling(3, min_periods=3).sum()
        out["up_gap_count"] = out["gap"].gt(0.0).astype(float).where(out["gap"].notna()).rolling(3, min_periods=3).sum()
        out["down_gap_count"] = out["gap"].lt(0.0).astype(float).where(out["gap"].notna()).rolling(3, min_periods=3).sum()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        scale = (out["return_volatility"] * np.sqrt(3.0)).replace(0.0, np.nan)
        out["tension"] = ((out["gap_sum"] - out["return_sum"]) / scale).replace([np.inf, -np.inf], np.nan)
        ready = np.isfinite(out[["return", "return_sum", "gap_sum", "tension", "atr"]]).all(axis=1) & out["atr"].gt(0.0)
        long_setup = ready & out["down_gap_count"].eq(3.0) & out["return_sum"].gt(0.0) & out["return"].gt(0.0) & out["tension"].lt(-params.tension_threshold)
        short_setup = ready & out["up_gap_count"].eq(3.0) & out["return_sum"].lt(0.0) & out["return"].lt(0.0) & out["tension"].gt(params.tension_threshold)
        out["candidate"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapFailureParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        direction = 0
        pending_entry = False
        fixed_stop = np.nan
        entry_atr = np.nan
        held_bars = 0

        for i in range(len(data)):
            if direction != 0:
                # Yesterday's decision fills at today's open; freeze its ATR.
                if pending_entry:
                    if not np.isfinite(opens[i]) or opens[i] <= 0.0:
                        direction = 0
                        pending_entry = False
                        continue
                    fixed_stop = opens[i] - direction * params.stop_atr * entry_atr
                    pending_entry = False
                    held_bars = 0
                held_bars += 1
                valid_close = np.isfinite(closes[i]) and closes[i] > 0.0
                stopped = valid_close and ((direction == 1 and closes[i] < fixed_stop) or (direction == -1 and closes[i] > fixed_stop))
                if not valid_close or stopped or held_bars >= 4:
                    direction = 0
                    # Leave a flat decision on exit; do not reopen this bar.
                decisions[i] = direction
            elif candidates[i] != 0 and np.isfinite(atr[i]) and atr[i] > 0.0:
                direction = int(candidates[i])
                entry_atr = atr[i]
                pending_entry = True
                decisions[i] = direction

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
