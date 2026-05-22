from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GenA2Params:
    roc_period: int = 10
    accel_lag: int = 5
    dd_lookback: int = 60
    dd_enter: float = 0.07
    dd_exit: float = 0.02
    accel_threshold: float = 0.0
    ma_period: int = 200
    profit_target: float = 0.06
    time_stop: int = 18


class GeneratedStrategy(BaseStrategy[GenA2Params]):
    strategy_id = "gen_a2_1779146249"

    @classmethod
    def params_type(cls) -> type[GenA2Params]:
        return GenA2Params

    @staticmethod
    def warmup_bars(params: GenA2Params) -> int:
        return int(max(params.ma_period, params.dd_lookback,
                       params.roc_period + params.accel_lag)) + 1

    def indicators(self, data: pd.DataFrame, params: GenA2Params) -> pd.DataFrame:
        close = data["close"]
        roc = close.pct_change(params.roc_period)
        accel = roc - roc.shift(params.accel_lag)
        peak = close.rolling(params.dd_lookback, min_periods=params.dd_lookback).max()
        trough = close.rolling(params.dd_lookback, min_periods=params.dd_lookback).min()
        drawdown = close / peak - 1.0
        drawup = close / trough - 1.0
        ma = close.rolling(params.ma_period, min_periods=params.ma_period).mean()
        return pd.DataFrame(
            {
                "accel": accel,
                "drawdown": drawdown,
                "drawup": drawup,
                "ma": ma,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GenA2Params,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        accel = indicators["accel"].to_numpy(dtype=float)
        drawdown = indicators["drawdown"].to_numpy(dtype=float)
        drawup = indicators["drawup"].to_numpy(dtype=float)
        ma = indicators["ma"].to_numpy(dtype=float)

        target = np.zeros(n, dtype=int)

        position = 0
        entry_price = 0.0
        bars_held = 0
        armed_long = False
        armed_short = False

        dd_enter = abs(params.dd_enter)
        dd_exit = abs(params.dd_exit)
        accel_thr = abs(params.accel_threshold)

        for i in range(n):
            dd = drawdown[i]
            du = drawup[i]
            ac = accel[i]
            ma_i = ma[i]
            px = close[i]

            valid = not (
                np.isnan(dd) or np.isnan(du) or np.isnan(ac) or np.isnan(ma_i)
            )
            if not valid:
                target[i] = position
                continue

            if position == 0:
                if dd <= -dd_enter:
                    armed_long = True
                elif dd >= -dd_exit:
                    armed_long = False

                if du >= dd_enter:
                    armed_short = True
                elif du <= dd_exit:
                    armed_short = False

                if armed_long and ac > accel_thr and px > ma_i:
                    position = 1
                    entry_price = px
                    bars_held = 0
                    armed_long = False
                    armed_short = False
                elif armed_short and ac < -accel_thr and px < ma_i:
                    position = -1
                    entry_price = px
                    bars_held = 0
                    armed_long = False
                    armed_short = False
            else:
                bars_held += 1
                if entry_price > 0.0:
                    pnl = (px - entry_price) / entry_price * position
                else:
                    pnl = 0.0
                if pnl >= params.profit_target or bars_held >= params.time_stop:
                    position = 0
                    entry_price = 0.0
                    bars_held = 0

            target[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
