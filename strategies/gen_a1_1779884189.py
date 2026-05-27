from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback_high: int = 20
    atr_period: int = 14
    drawdown_atr_k: float = 2.0
    susceptible_window: int = 10
    susceptible_min: int = 3
    trail_atr_k: float = 2.5
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_a1_1779884189"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.lookback_high, params.atr_period, params.susceptible_window) + 2

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        rolling_high = close.rolling(params.lookback_high, min_periods=params.lookback_high).max()
        drawdown = rolling_high - close
        safe_atr = atr.where(atr > 0)
        dd_atr = drawdown / safe_atr

        deep_dd = (dd_atr >= params.drawdown_atr_k).astype(float)
        susceptible_count = deep_dd.rolling(params.susceptible_window, min_periods=params.susceptible_window).sum()

        ret = close.pct_change()
        up_bar = (ret > 0).astype(float)
        two_up = up_bar.rolling(2, min_periods=2).sum()

        out = pd.DataFrame({
            "atr": atr,
            "rolling_high": rolling_high,
            "dd_atr": dd_atr,
            "deep_dd": deep_dd,
            "susceptible_count": susceptible_count,
            "two_up": two_up,
        }, index=data.index)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        susceptible = indicators["susceptible_count"].to_numpy()
        two_up = indicators["two_up"].to_numpy()
        dd_atr = indicators["dd_atr"].to_numpy()

        n = len(close)
        sig = np.zeros(n, dtype=np.int64)
        size = np.full(n, 1.0, dtype=np.float64)

        in_pos = False
        hwm = 0.0
        hold = 0

        for i in range(n):
            if not in_pos:
                if (
                    not np.isnan(susceptible[i])
                    and not np.isnan(two_up[i])
                    and not np.isnan(dd_atr[i])
                    and susceptible[i] >= params.susceptible_min
                    and two_up[i] >= 2.0
                ):
                    sig[i] = 1
                    in_pos = True
                    hwm = close[i]
                    hold = 0
            else:
                hold += 1
                if close[i] > hwm:
                    hwm = close[i]
                atr_i = atr[i] if not np.isnan(atr[i]) else 0.0
                stop_dist = params.trail_atr_k * atr_i
                exit_now = False
                if stop_dist > 0 and close[i] <= hwm - stop_dist:
                    exit_now = True
                if hold >= params.max_hold_bars:
                    exit_now = True
                if exit_now:
                    sig[i] = 0
                    in_pos = False
                    hwm = 0.0
                    hold = 0
                else:
                    sig[i] = 1

        df = pd.DataFrame({"signal": sig, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
