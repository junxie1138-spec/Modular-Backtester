from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    ma_period: int = 200
    runs_window: int = 30
    runs_z_threshold: float = 0.5
    min_streak: int = 2
    profit_target: float = 0.04
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1778914657"

    @classmethod
    def params_type(cls) -> type[GeneratedParams]:
        return GeneratedParams

    @staticmethod
    def warmup_bars(params: GeneratedParams) -> int:
        return int(max(params.ma_period, params.runs_window)) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        ind = pd.DataFrame(index=data.index)

        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        ind["ma"] = ma
        ind["close"] = close

        # up-close boolean and its integer form
        up = close > close.shift(1)
        up_int = up.astype(int)

        # consecutive up-close streak length ending at each bar
        grp = (up_int == 0).cumsum()
        streak = (up_int.groupby(grp).cumcount() + 1) * up_int
        ind["streak"] = streak.astype(float)

        # Wald-Wolfowitz runs test on the last runs_window sign sequence.
        # runs = 1 + number of adjacent sign changes among the W bars (W-1 pairs).
        W = int(params.runs_window)
        change = (up_int != up_int.shift(1)).astype(int)
        runs = 1.0 + change.rolling(W - 1, min_periods=W - 1).sum()

        n_pos = up_int.rolling(W, min_periods=W).sum().astype(float)
        n_neg = float(W) - n_pos
        N = float(W)

        prod = n_pos * n_neg
        mean_r = 2.0 * prod / N + 1.0
        var_r = (2.0 * prod * (2.0 * prod - N)) / (N * N * (N - 1.0))
        var_r = var_r.where(var_r > 1e-12, np.nan)
        runs_z = (runs - mean_r) / np.sqrt(var_r)
        ind["runs_z"] = runs_z

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        idx = data.index
        n = len(idx)

        close = indicators["close"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)
        runs_z = indicators["runs_z"].to_numpy(dtype=float)
        streak = indicators["streak"].to_numpy(dtype=float)

        # Entry: fewer runs than a coin-flip null (positive-autocorrelation
        # regime), a live up-close streak, and a bull regime above the 200MA.
        entry = (
            np.isfinite(runs_z)
            & np.isfinite(ma)
            & np.isfinite(streak)
            & (runs_z < -float(params.runs_z_threshold))
            & (streak >= float(params.min_streak))
            & (close > ma)
        )

        pt = float(params.profit_target)
        max_hold = int(params.max_hold_bars)

        pos = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                hit_pt = close[i] >= entry_price * (1.0 + pt)
                hit_time = bars_held >= max_hold
                if hit_pt or hit_time:
                    in_pos = False
                    pos[i] = 0
                else:
                    pos[i] = 1
            else:
                if entry[i]:
                    in_pos = True
                    entry_price = close[i]
                    bars_held = 0
                    pos[i] = 1
                else:
                    pos[i] = 0

        df = pd.DataFrame(index=idx)
        df["signal"] = pos
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
