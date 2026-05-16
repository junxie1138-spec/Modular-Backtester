from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 20
    pct_window: int = 120
    entry_pct: float = 0.75
    vol_window: int = 20
    target_vol: float = 0.15
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778897253"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.peak_window, params.pct_window, params.vol_window) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]

        rolling_peak = close.rolling(params.peak_window).max()
        safe_peak = rolling_peak.replace(0, np.nan)
        drawdown_depth = (close - safe_peak) / safe_peak
        abs_drawdown = drawdown_depth.abs()

        # Rolling percentile rank of absolute drawdown depth.
        # rank=1.0 means current drawdown is the deepest seen in the window.
        drawdown_pct_rank = abs_drawdown.rolling(params.pct_window).rank(pct=True)

        log_ret = np.log(close / close.shift(1))
        realized_vol = log_ret.rolling(params.vol_window).std() * np.sqrt(252)

        ind = pd.DataFrame(index=data.index)
        ind["drawdown_pct_rank"] = drawdown_pct_rank
        ind["realized_vol"] = realized_vol
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size_arr = np.full(n, 0.95)

        dpct = indicators["drawdown_pct_rank"].to_numpy()
        rvol = indicators["realized_vol"].to_numpy()

        in_trade_until = -1
        entry_size = 0.95

        for i in range(n):
            if np.isnan(dpct[i]) or np.isnan(rvol[i]) or rvol[i] <= 0.0:
                continue

            if i <= in_trade_until:
                signal[i] = 1
                size_arr[i] = entry_size
                continue

            if dpct[i] >= params.entry_pct:
                signal[i] = 1
                raw_size = params.target_vol / rvol[i]
                entry_size = float(np.clip(raw_size, 0.05, 1.0))
                size_arr[i] = entry_size
                in_trade_until = i + params.hold_bars - 1

        df = pd.DataFrame({"signal": signal, "size": size_arr}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).clip(lower=0.01)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
