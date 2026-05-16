from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    svf_window: int = 10
    svf_acc_window: int = 5
    svf_acc_threshold: float = 0.02
    ma_period: int = 200
    atr_period: int = 14
    breakeven_pct: float = 0.015
    trail_k: float = 2.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778906914"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.ma_period + params.svf_window + params.svf_acc_window

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        ind["ma200"] = data["close"].rolling(params.ma_period).mean()
        ind["bull_regime"] = (data["close"] > ind["ma200"]).astype(float)

        signed_vol = (
            np.sign(data["close"] - data["open"])
            * (data["high"] - data["low"])
            / data["close"]
        )
        ind["svf"] = signed_vol.rolling(params.svf_window).sum()
        ind["svf_acc"] = ind["svf"] - ind["svf"].shift(params.svf_acc_window)

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].values
        atr = indicators["atr"].values
        svf = indicators["svf"].values
        svf_acc = indicators["svf_acc"].values
        bull = indicators["bull_regime"].values
        ma200 = indicators["ma200"].values
        thr = params.svf_acc_threshold

        valid = ~(
            np.isnan(svf)
            | np.isnan(svf_acc)
            | np.isnan(atr)
            | np.isnan(ma200)
        )
        raw_long = valid & (svf_acc > thr) & (svf > 0.0) & (bull == 1.0)
        raw_short = valid & (svf_acc < -thr) & (svf < 0.0) & (bull == 0.0)
        raw_entry = np.where(raw_long, 1, np.where(raw_short, -1, 0)).astype(int)

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(raw_entry, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0

        entry = df["signal"].values.copy()
        final_signal = np.zeros(n, dtype=int)
        position = 0
        entry_price = 0.0
        stop_level = 0.0
        be_triggered = False

        for i in range(n):
            curr_atr = atr[i] if not np.isnan(atr[i]) else close[i] * 0.01

            if position == 0:
                if entry[i] == 1:
                    position = 1
                    entry_price = close[i]
                    stop_level = entry_price - params.trail_k * curr_atr
                    be_triggered = False
                elif entry[i] == -1:
                    position = -1
                    entry_price = close[i]
                    stop_level = entry_price + params.trail_k * curr_atr
                    be_triggered = False

            if position == 1:
                if not be_triggered and close[i] >= entry_price * (1.0 + params.breakeven_pct):
                    stop_level = max(stop_level, entry_price)
                    be_triggered = True
                new_stop = close[i] - params.trail_k * curr_atr
                if new_stop > stop_level:
                    stop_level = new_stop
                if close[i] <= stop_level:
                    position = 0
                else:
                    final_signal[i] = 1

            elif position == -1:
                if not be_triggered and close[i] <= entry_price * (1.0 - params.breakeven_pct):
                    stop_level = min(stop_level, entry_price)
                    be_triggered = True
                new_stop = close[i] + params.trail_k * curr_atr
                if new_stop < stop_level:
                    stop_level = new_stop
                if close[i] >= stop_level:
                    position = 0
                else:
                    final_signal[i] = -1

        df["signal"] = final_signal
        return SignalFrame(data=df, signal_column="signal", size_column="size")
