from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TidalNodeParams:
    rank_window: int = 5
    rank_threshold: float = 0.25
    entry_weekday: int = 0
    profit_target_pct: float = 0.015
    time_stop_bars: int = 4


class GeneratedStrategy(BaseStrategy[TidalNodeParams]):
    strategy_id = "gen_a1_1779882689"

    @classmethod
    def params_type(cls) -> type[TidalNodeParams]:
        return TidalNodeParams

    def warmup_bars(self, params: TidalNodeParams) -> int:
        return max(int(params.rank_window) + 1, 6)

    def indicators(self, data: pd.DataFrame, params: TidalNodeParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        w = max(2, int(params.rank_window))
        out["close_rank"] = data["close"].rolling(w, min_periods=w).rank(pct=True)
        if isinstance(data.index, pd.DatetimeIndex):
            out["weekday"] = np.asarray(data.index.weekday, dtype=np.int64)
        else:
            out["weekday"] = np.zeros(len(data), dtype=np.int64)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TidalNodeParams,
    ) -> SignalFrame:
        n = len(data)
        df = pd.DataFrame(index=data.index)
        raw_signal = np.zeros(n, dtype=np.int64)

        close = data["close"].to_numpy(dtype=np.float64, copy=False)
        rank = indicators["close_rank"].to_numpy(dtype=np.float64, copy=False)
        weekday = np.asarray(indicators["weekday"].to_numpy(), dtype=np.int64)

        target_wd = int(params.entry_weekday) % 7
        thr = float(params.rank_threshold)
        ptgt = float(params.profit_target_pct)
        tstop = max(1, int(params.time_stop_bars))

        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            r = rank[i]
            c = close[i]
            wd = weekday[i] if i < weekday.shape[0] else -1

            if in_pos:
                bars_held += 1
                exit_now = False
                if entry_price > 0.0 and c >= entry_price * (1.0 + ptgt):
                    exit_now = True
                if bars_held >= tstop:
                    exit_now = True
                if exit_now:
                    raw_signal[i] = 0
                    in_pos = False
                    bars_held = 0
                    entry_price = 0.0
                else:
                    raw_signal[i] = 1
            else:
                if (not np.isnan(r)) and (wd == target_wd) and (r <= thr) and not np.isnan(c):
                    in_pos = True
                    entry_price = c
                    bars_held = 0
                    raw_signal[i] = 1

        signal_series = pd.Series(raw_signal, index=data.index, dtype=np.int64)
        df["signal"] = signal_series.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
