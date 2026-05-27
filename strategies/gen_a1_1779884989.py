from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fast_window: int = 5
    slow_window: int = 20
    divergence_threshold: float = 0.20
    confirm_bars: int = 2
    atr_window: int = 14
    breakeven_pct: float = 0.005
    atr_trail_mult: float = 2.0
    spike_z: float = 2.5
    spike_window: int = 20
    refractory_bars: int = 3


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779884989"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(max(
            params.slow_window + params.fast_window + 5,
            params.atr_window + 2,
            params.spike_window + 2,
            50,
        ))

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        ret = close.pct_change()
        ret_lag = ret.shift(1)

        fast_ac = ret.rolling(params.fast_window).corr(ret_lag)
        slow_ac = ret.rolling(params.slow_window).corr(ret_lag)
        divergence = fast_ac - slow_ac

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        ret_mean = ret.rolling(params.spike_window).mean()
        ret_std = ret.rolling(params.spike_window).std()
        z = (ret - ret_mean) / ret_std.replace(0, np.nan)

        return pd.DataFrame({
            "ret": ret,
            "fast_ac": fast_ac,
            "slow_ac": slow_ac,
            "divergence": divergence,
            "atr": atr,
            "z": z,
        }, index=data.index)

    @classmethod
    def generate_signals(cls, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        divergence = indicators["divergence"].to_numpy()
        z = indicators["z"].to_numpy()

        n = len(close)

        thr = float(params.divergence_threshold)
        cb = int(params.confirm_bars)
        bk = float(params.breakeven_pct)
        tm = float(params.atr_trail_mult)

        single_long = np.where(np.isnan(divergence), False, divergence > thr)
        single_short = np.where(np.isnan(divergence), False, divergence < -thr)

        confirm_long = (pd.Series(single_long.astype(np.float64))
                        .rolling(cb).sum().to_numpy() == cb)
        confirm_short = (pd.Series(single_short.astype(np.float64))
                         .rolling(cb).sum().to_numpy() == cb)

        spike = np.where(np.isnan(z), False, np.abs(z) > float(params.spike_z))
        refractory = (pd.Series(spike.astype(np.float64))
                      .rolling(int(params.refractory_bars)).sum().to_numpy() > 0)

        position = 0
        entry_price = 0.0
        stop_price = 0.0
        breakeven_armed = False

        raw_signal = np.zeros(n, dtype=np.int8)

        for i in range(n):
            price = close[i]
            a = atr[i]

            if position == 0:
                if i < cb:
                    continue
                if refractory[i]:
                    continue
                if confirm_long[i]:
                    position = 1
                    entry_price = price
                    stop_price = (price - tm * a) if not np.isnan(a) else price * (1.0 - 0.02)
                    breakeven_armed = False
                    raw_signal[i] = 1
                elif confirm_short[i]:
                    position = -1
                    entry_price = price
                    stop_price = (price + tm * a) if not np.isnan(a) else price * (1.0 + 0.02)
                    breakeven_armed = False
                    raw_signal[i] = -1

            elif position == 1:
                if (not breakeven_armed) and price >= entry_price * (1.0 + bk):
                    if entry_price > stop_price:
                        stop_price = entry_price
                    breakeven_armed = True
                if breakeven_armed and (not np.isnan(a)):
                    new_stop = price - tm * a
                    if new_stop > stop_price:
                        stop_price = new_stop
                if price <= stop_price:
                    position = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1

            else:
                if (not breakeven_armed) and price <= entry_price * (1.0 - bk):
                    if entry_price < stop_price:
                        stop_price = entry_price
                    breakeven_armed = True
                if breakeven_armed and (not np.isnan(a)):
                    new_stop = price + tm * a
                    if new_stop < stop_price:
                        stop_price = new_stop
                if price >= stop_price:
                    position = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = -1

        df = pd.DataFrame({"signal": raw_signal}, index=data.index)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
