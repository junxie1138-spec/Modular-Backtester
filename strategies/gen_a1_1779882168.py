from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenAutocorrRankParams:
    ac_window: int = 10
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GenAutocorrRankParams]):
    strategy_id = "gen_a1_1779882168"

    @classmethod
    def params_type(cls):
        return GenAutocorrRankParams

    def warmup_bars(self, params: GenAutocorrRankParams) -> int:
        return 126 + max(int(params.ac_window), 1) + 30

    def indicators(self, data: pd.DataFrame, params: GenAutocorrRankParams) -> pd.DataFrame:
        close = data["close"]
        returns = close.pct_change()

        ac_window = max(int(params.ac_window), 3)
        lagged = returns.shift(1)
        ac_lag1 = returns.rolling(ac_window).corr(lagged)

        rank_window = 126
        ac_rank = ac_lag1.rolling(rank_window).rank(pct=True)

        std_window = 20
        rolling_std = returns.rolling(std_window).std()
        down_spike = (returns < -2.0 * rolling_std)

        out = pd.DataFrame(index=data.index)
        out["returns"] = returns
        out["ac_lag1"] = ac_lag1
        out["ac_rank"] = ac_rank
        out["down_spike"] = down_spike.astype(float)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenAutocorrRankParams,
    ) -> SignalFrame:
        n = len(data)
        close_arr = data["close"].to_numpy(dtype=float)
        ac_rank = indicators["ac_rank"].to_numpy(dtype=float)
        down_spike_arr = indicators["down_spike"].to_numpy(dtype=float) > 0.5

        ac_low = np.where(np.isnan(ac_rank), False, ac_rank <= 0.15)
        entry_cond = down_spike_arr & ac_low

        hold = max(int(params.hold_bars), 1)
        refractory = hold
        profit_target = 0.03

        raw = np.zeros(n, dtype=np.int64)
        in_pos = False
        entry_price = 0.0
        entry_idx = -1
        last_exit_idx = -(10 ** 9)

        for i in range(n):
            cur = close_arr[i]
            if in_pos:
                bars_held = i - entry_idx
                hit_target = (entry_price > 0.0) and (cur / entry_price - 1.0) >= profit_target
                if hit_target or bars_held >= hold:
                    raw[i] = 0
                    in_pos = False
                    entry_price = 0.0
                    entry_idx = -1
                    last_exit_idx = i
                else:
                    raw[i] = 1
            else:
                if (i - last_exit_idx) > refractory and bool(entry_cond[i]) and not np.isnan(cur):
                    in_pos = True
                    entry_price = cur
                    entry_idx = i
                    raw[i] = 1
                else:
                    raw[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
