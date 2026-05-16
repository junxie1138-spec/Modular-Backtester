from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ema_period: int = 50
    zscore_period: int = 50
    zscore_thresh: float = 1.2
    skew_period: int = 20
    skew_lag: int = 5
    skew_neg_thresh: float = 0.2
    atr_period: int = 14
    breakeven_pct: float = 0.015
    trail_mult: float = 2.0


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778893003"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.zscore_period, params.skew_period) + params.skew_lag + params.atr_period + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ema = close.ewm(span=params.ema_period, adjust=False).mean()
        rolling_std = close.rolling(params.zscore_period).std()
        ind["zscore"] = (close - ema) / rolling_std.replace(0.0, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        ret = close.pct_change()
        ind["skew"] = ret.rolling(params.skew_period).skew()
        ind["skew_lag"] = ind["skew"].shift(params.skew_lag)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        zscore = indicators["zscore"].values
        atr = indicators["atr"].values
        skew = indicators["skew"].values
        skew_lag_vals = indicators["skew_lag"].values

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_trade = False
        entry_price = 0.0
        stop_level = 0.0
        breakeven_reached = False

        for i in range(n):
            z = zscore[i]
            a = atr[i]
            sk = skew[i]
            sk_lag = skew_lag_vals[i]

            if in_trade:
                if not np.isnan(a):
                    candidate_stop = close[i] - params.trail_mult * a
                    if not breakeven_reached:
                        if close[i] >= entry_price * (1.0 + params.breakeven_pct):
                            breakeven_reached = True
                            stop_level = max(stop_level, entry_price)
                    if breakeven_reached:
                        stop_level = max(stop_level, candidate_stop)

                if close[i] <= stop_level:
                    raw_signal[i] = 0
                    in_trade = False
                    entry_price = 0.0
                    stop_level = 0.0
                    breakeven_reached = False
                else:
                    raw_signal[i] = 1
            else:
                if (
                    not np.isnan(z)
                    and not np.isnan(sk)
                    and not np.isnan(sk_lag)
                    and not np.isnan(a)
                    and z < -params.zscore_thresh
                    and sk > sk_lag
                    and sk_lag < -params.skew_neg_thresh
                ):
                    raw_signal[i] = 1
                    in_trade = True
                    entry_price = close[i]
                    stop_level = entry_price - params.trail_mult * a
                    breakeven_reached = False
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame({"signal": raw_signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
