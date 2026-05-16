from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_period: int = 5
    shock_min_bars: int = 3
    hold_bars: int = 4
    vol_window: int = 20
    vol_target: float = 0.12


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778911552"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(200, params.vol_window + params.roc_period + 3)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]

        ma200 = close.rolling(200, min_periods=200).mean()

        roc = close.pct_change(params.roc_period)
        roc_accel = roc.diff()

        daily_ret = close.pct_change()
        realized_vol = (
            daily_ret.rolling(params.vol_window, min_periods=params.vol_window).std()
            * np.sqrt(252)
        )

        ind = pd.DataFrame(index=data.index)
        ind["ma200"] = ma200
        ind["roc_accel"] = roc_accel
        ind["realized_vol"] = realized_vol
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close_arr = data["close"].to_numpy(dtype=np.float64)
        ma200 = indicators["ma200"].to_numpy(dtype=np.float64)
        roc_accel = indicators["roc_accel"].to_numpy(dtype=np.float64)
        realized_vol = indicators["realized_vol"].to_numpy(dtype=np.float64)
        n = len(close_arr)

        # Consecutive bars of negative ROC acceleration (shockwave buildup counter)
        consec_decel = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            ra = roc_accel[i]
            if not np.isfinite(ra) or ra >= 0.0:
                consec_decel[i] = 0
            else:
                consec_decel[i] = consec_decel[i - 1] + 1

        signal = np.zeros(n, dtype=np.int32)
        size = np.full(n, 0.95, dtype=np.float64)

        i = 1
        while i < n:
            if (
                not np.isfinite(ma200[i])
                or not np.isfinite(roc_accel[i])
                or not np.isfinite(realized_vol[i])
            ):
                i += 1
                continue

            regime_ok = close_arr[i] > ma200[i]
            accel_positive = roc_accel[i] > 0.0
            buildup_ok = consec_decel[i - 1] >= params.shock_min_bars

            if regime_ok and accel_positive and buildup_ok:
                rv = realized_vol[i]
                pos_size = min(0.95, params.vol_target / rv) if rv > 1e-9 else 0.95
                exit_bar = min(i + params.hold_bars, n)
                for j in range(i, exit_bar):
                    signal[j] = 1
                    size[j] = pos_size
                i = exit_bar
            else:
                i += 1

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
