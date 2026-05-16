from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FlushParams:
    lookback: int = 5
    rank_window: int = 63
    low_pctl: float = 0.10
    high_pctl: float = 0.85
    atr_window: int = 14
    atr_k: float = 2.5
    max_hold: int = 18
    refractory: int = 5


class GeneratedStrategy(BaseStrategy[FlushParams]):
    strategy_id = "gen_a1_1778906007"

    @classmethod
    def params_type(cls) -> type[FlushParams]:
        return FlushParams

    @staticmethod
    def warmup_bars(params: FlushParams) -> int:
        return int(max(params.lookback + params.rank_window, params.atr_window + 1) + 5)

    def indicators(self, data: pd.DataFrame, params: FlushParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Cumulative multi-bar return, then its trailing percentile rank.
        ret_n = close.pct_change(params.lookback)
        out["ret_rank"] = ret_n.rolling(params.rank_window).rank(pct=True)

        # Volume percentile rank within the same trailing window.
        out["vol_rank"] = volume.rolling(params.rank_window).rank(pct=True)

        # ATR for the fixed volatility stop.
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(params.atr_window).mean()

        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: FlushParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)

        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ret_rank = indicators["ret_rank"].to_numpy(dtype=float)
        vol_rank = indicators["vol_rank"].to_numpy(dtype=float)
        n = len(close)

        signal = np.zeros(n, dtype=int)

        in_pos = False
        entry_stop = 0.0
        bars_held = 0
        last_entry = -(10 ** 9)

        for i in range(n):
            if in_pos:
                bars_held += 1
                exit_now = False
                # Fixed volatility stop: entry price minus k*ATR, set once at entry.
                if not np.isnan(close[i]) and close[i] < entry_stop:
                    exit_now = True
                elif bars_held >= params.max_hold:
                    exit_now = True
                if exit_now:
                    in_pos = False
                    signal[i] = 0
                else:
                    signal[i] = 1
            else:
                entry_ok = (
                    not np.isnan(ret_rank[i])
                    and not np.isnan(vol_rank[i])
                    and not np.isnan(atr[i])
                    and atr[i] > 0.0
                    and ret_rank[i] <= params.low_pctl
                    and vol_rank[i] >= params.high_pctl
                    and (i - last_entry) > params.refractory
                )
                if entry_ok:
                    in_pos = True
                    entry_stop = close[i] - params.atr_k * atr[i]
                    bars_held = 0
                    last_entry = i
                    signal[i] = 1
                else:
                    signal[i] = 0

        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
