from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    carve_window: int = 20
    thrust_lookback: int = 5
    atr_window: int = 14
    arm_z_in: float = 1.5
    arm_z_out: float = 0.5
    initial_stop_k: float = 2.5
    be_trigger_pct: float = 0.02
    atr_trail_k: float = 2.0
    ma_filter: int = 200


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779881396"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(max(params.ma_filter, params.atr_window, params.carve_window)
                   + params.thrust_lookback + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        upper_carve = (high - prev_high).clip(lower=0.0)
        lower_carve = (prev_low - low).clip(lower=0.0)

        upper_thrust = upper_carve.rolling(
            params.thrust_lookback, min_periods=params.thrust_lookback
        ).sum()
        lower_erosion = lower_carve.rolling(
            params.thrust_lookback, min_periods=params.thrust_lookback
        ).sum()

        atr_safe = atr.where(atr > 0.0, np.nan)
        thrust_score = (upper_thrust - lower_erosion) / atr_safe

        ma = close.rolling(params.ma_filter, min_periods=params.ma_filter).mean()

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["thrust_score"] = thrust_score
        ind["ma"] = ma
        ind["upper_thrust"] = upper_thrust
        ind["lower_erosion"] = lower_erosion
        return ind

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        thrust = indicators["thrust_score"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)

        n = len(close)
        raw_signal = np.zeros(n, dtype=np.int64)

        armed = False
        in_pos = False
        entry_price = 0.0
        stop = -np.inf
        be_armed = False
        in_trade_high = 0.0

        for i in range(n):
            valid = (
                not (np.isnan(thrust[i]) or np.isnan(atr[i]) or np.isnan(ma[i]))
                and atr[i] > 0.0
            )

            if not valid:
                if in_pos:
                    raw_signal[i] = 1
                armed = False
                continue

            armed_prev_local = armed
            above_ma = bool(close[i] > ma[i])

            if not armed:
                if thrust[i] >= params.arm_z_in and above_ma:
                    armed = True
            else:
                if thrust[i] <= params.arm_z_out or not above_ma:
                    armed = False

            if in_pos:
                if close[i] > in_trade_high:
                    in_trade_high = close[i]

                if not be_armed and close[i] >= entry_price * (1.0 + params.be_trigger_pct):
                    be_armed = True
                    if entry_price > stop:
                        stop = entry_price

                if be_armed:
                    candidate = in_trade_high - params.atr_trail_k * atr[i]
                    if candidate > stop:
                        stop = candidate

                if close[i] <= stop:
                    raw_signal[i] = 0
                    in_pos = False
                    entry_price = 0.0
                    stop = -np.inf
                    be_armed = False
                    in_trade_high = 0.0
                else:
                    raw_signal[i] = 1
            else:
                if armed and armed_prev_local:
                    raw_signal[i] = 1
                    in_pos = True
                    entry_price = float(close[i])
                    in_trade_high = float(close[i])
                    stop = entry_price - params.initial_stop_k * atr[i]
                    be_armed = False

        sig = pd.Series(raw_signal, index=data.index, dtype=np.int64)
        sig = sig.shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index, dtype=float)

        df = pd.DataFrame({"signal": sig, "size": size}, index=data.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
