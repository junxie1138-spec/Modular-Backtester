from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 20
    dd_threshold: float = 0.03


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778895960"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        # prevalence needs lb bars; it is built on infected which needs lb bars: 2*lb + diff
        return params.lookback * 2 + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        lb = params.lookback
        thr = params.dd_threshold

        roll_high = close.rolling(lb).max()
        roll_low = close.rolling(lb).min()

        drawdown = (close - roll_high) / roll_high.replace(0, np.nan)
        updrift = (close - roll_low) / roll_low.replace(0, np.nan)

        # Binary infection flags
        infected_down = (drawdown < -thr).astype(float)
        infected_up = (updrift > thr).astype(float)

        # Epidemic prevalence: rolling fraction of infected bars
        prevalence_down = infected_down.rolling(lb).mean()
        prevalence_up = infected_up.rolling(lb).mean()

        # First difference of prevalence = epidemic trajectory
        d_prev_down = prevalence_down.diff()
        d_prev_up = prevalence_up.diff()

        ind = pd.DataFrame(index=data.index)
        ind["drawdown"] = drawdown
        ind["updrift"] = updrift
        ind["prevalence_down"] = prevalence_down
        ind["prevalence_up"] = prevalence_up
        ind["d_prev_down"] = d_prev_down
        ind["d_prev_up"] = d_prev_up
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        thr = params.dd_threshold

        # Long: currently infected (in drawdown), high prevalence, epidemic turning down
        long_cond = (
            (indicators["drawdown"] < -thr)
            & (indicators["prevalence_down"] > 0.4)
            & (indicators["d_prev_down"] < 0)
        ).fillna(False)

        # Short: currently elevated (above trough), high prevalence, epidemic turning down
        short_cond = (
            (indicators["updrift"] > thr)
            & (indicators["prevalence_up"] > 0.4)
            & (indicators["d_prev_up"] < 0)
        ).fillna(False)

        long_arr = long_cond.to_numpy()
        short_arr = short_cond.to_numpy()
        n = len(long_arr)

        raw_signals = np.zeros(n, dtype=int)
        sig = 0
        for i in range(n):
            # Signal-reversal exit: hold until condition flips
            if sig == 1 and not long_arr[i]:
                sig = 0
            elif sig == -1 and not short_arr[i]:
                sig = 0
            # Enter when flat and condition fires
            if sig == 0:
                if long_arr[i]:
                    sig = 1
                elif short_arr[i]:
                    sig = -1
            raw_signals[i] = sig

        dd_abs = indicators["drawdown"].abs().fillna(0.0).to_numpy()
        ud_abs = indicators["updrift"].abs().fillna(0.0).to_numpy()
        size_arr = np.where(
            raw_signals == 1,
            np.clip(1.0 + dd_abs * 10.0, 1.0, 3.0),
            np.where(
                raw_signals == -1,
                np.clip(1.0 + ud_abs * 10.0, 1.0, 3.0),
                1.0,
            ),
        )

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signals
        df["size"] = size_arr

        # Mandatory 1-bar lookahead shift
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
