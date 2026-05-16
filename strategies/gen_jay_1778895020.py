from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    acf_window: int = 21
    lag_front: int = 4
    entry_threshold: float = 0.02
    front_threshold: float = 0.03
    size_multiplier: float = 3.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778895020"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.acf_window + params.lag_front + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        w = params.acf_window
        lf = params.lag_front

        ret_s1 = ret.shift(1)
        ret_sf = ret.shift(lf)

        mean_r = ret.rolling(w).mean()
        std_r = ret.rolling(w).std()

        mean_s1 = ret_s1.rolling(w).mean()
        std_s1 = ret_s1.rolling(w).std()
        cov1 = (ret * ret_s1).rolling(w).mean() - mean_r * mean_s1
        acf1 = cov1 / (std_r * std_s1 + 1e-12)

        mean_sf = ret_sf.rolling(w).mean()
        std_sf = ret_sf.rolling(w).std()
        cov_f = (ret * ret_sf).rolling(w).mean() - mean_r * mean_sf
        acf_front = cov_f / (std_r * std_sf + 1e-12)

        ind = pd.DataFrame(index=data.index)
        ind["acf1"] = acf1
        ind["acf_front"] = acf_front
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        acf1_vals = indicators["acf1"].values
        acf_front_vals = indicators["acf_front"].values
        n = len(data)

        raw_sig = np.zeros(n, dtype=int)
        raw_sz = np.full(n, 0.1)

        in_long = False

        for i in range(n):
            a1 = acf1_vals[i]
            af = acf_front_vals[i]

            if np.isnan(a1) or np.isnan(af):
                continue

            on = (a1 > params.entry_threshold) and (af > params.front_threshold)

            if not in_long and on:
                in_long = True
            elif in_long and not on:
                in_long = False

            if in_long:
                raw_sig[i] = 1
                raw_sz[i] = float(np.clip(a1 * params.size_multiplier, 0.1, 1.0))

        df = pd.DataFrame({"signal": raw_sig, "size": raw_sz}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.1)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
