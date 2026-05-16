from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    dd_window: int = 20
    rank_window: int = 60
    atr_period: int = 14
    k_atr: float = 2.0
    deep_z: float = 1.0
    shallow_z: float = 0.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778906120"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.dd_window, params.rank_window, params.atr_period) + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        rolling_high = close.rolling(params.dd_window).max()
        drawdown = (close - rolling_high) / rolling_high.where(rolling_high > 0, np.nan)

        # First difference of drawdown: positive = price recovering toward rolling high
        dd_velocity = drawdown.diff()

        dd_mean = drawdown.rolling(params.rank_window).mean()
        dd_std = drawdown.rolling(params.rank_window).std()
        # NaN where std is near zero (flat price) to suppress false signals
        dd_zscore = (drawdown - dd_mean) / dd_std.where(dd_std > 1e-10, np.nan)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=params.atr_period, adjust=False).mean()

        # Predator exhaustion: deep drawdown AND velocity turning positive (sellers losing grip)
        pred_exhaustion = (
            (dd_zscore < -params.deep_z) & (dd_velocity > 0)
        ).fillna(False)

        # Predator emergence: drawdown z-score elevated (price near/above mean relative high)
        # AND velocity turning negative (sellers gaining momentum from extended price)
        pred_emergence = (
            (dd_zscore > params.shallow_z) & (dd_velocity < 0)
        ).fillna(False)

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["pred_exhaustion"] = pred_exhaustion.astype(float)
        ind["pred_emergence"] = pred_emergence.astype(float)
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr_vals = indicators["atr"].values
        pred_ex = indicators["pred_exhaustion"].values
        pred_em = indicators["pred_emergence"].values

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)

        position = 0
        entry_price = 0.0
        entry_atr = 0.0

        for i in range(2, n):
            if position == 0:
                # Two-bar confirmation: both i-1 and i-2 must fire the setup signal
                if pred_ex[i - 1] == 1.0 and pred_ex[i - 2] == 1.0:
                    raw_signal[i] = 1
                    position = 1
                    entry_price = close[i]
                    cur_atr = atr_vals[i]
                    entry_atr = cur_atr if not np.isnan(cur_atr) else close[i] * 0.01
                elif pred_em[i - 1] == 1.0 and pred_em[i - 2] == 1.0:
                    raw_signal[i] = -1
                    position = -1
                    entry_price = close[i]
                    cur_atr = atr_vals[i]
                    entry_atr = cur_atr if not np.isnan(cur_atr) else close[i] * 0.01
            elif position == 1:
                # Fixed volatility stop: exit if close breaches entry minus k*ATR
                if close[i] <= entry_price - params.k_atr * entry_atr:
                    raw_signal[i] = 0
                    position = 0
                else:
                    raw_signal[i] = 1
            else:  # position == -1
                # Fixed volatility stop for short: exit if close exceeds entry plus k*ATR
                if close[i] >= entry_price + params.k_atr * entry_atr:
                    raw_signal[i] = 0
                    position = 0
                else:
                    raw_signal[i] = -1

        df = pd.DataFrame(
            {"signal": raw_signal, "size": np.ones(n, dtype=float)},
            index=data.index,
        )
        # Mandatory shift: bar-N decision fills on bar-N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
