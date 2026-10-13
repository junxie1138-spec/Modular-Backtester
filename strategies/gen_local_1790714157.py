from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 63
    drawdown_threshold: float = 0.08
    recovery_floor: float = 0.02
    atr_window: int = 14
    stop_atr: float = 2.0
    max_holding_bars: int = 10


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714157"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.peak_window, params.atr_window) + 1

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        close = data["close"]
        peak = close.rolling(params.peak_window).max()
        drawdown = close / peak - 1.0
        contraction = (drawdown > drawdown.shift(1)) & (drawdown.shift(1) > drawdown.shift(2))
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - close.shift(1)).abs(),
                (data["low"] - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()
        return pd.DataFrame(
            {"drawdown": drawdown, "contraction": contraction.fillna(False), "atr": atr},
            index=data.index,
        )

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        drawdown = indicators["drawdown"]
        setup = (
            (drawdown <= -params.drawdown_threshold)
            & indicators["contraction"]
        ).fillna(False)
        signal = pd.Series(0, index=data.index, dtype="int64")
        in_trade = False
        entry_price = 0.0
        entry_atr = 0.0
        held = 0

        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = indicators["atr"].iloc[i]
            if in_trade:
                held += 1
                stopped = close <= entry_price - params.stop_atr * entry_atr
                recovered = pd.notna(drawdown.iloc[i]) and drawdown.iloc[i] >= -params.recovery_floor
                if stopped or recovered or held >= params.max_holding_bars:
                    in_trade = False
                else:
                    signal.iloc[i] = 1
                    continue
            if not in_trade and bool(setup.iloc[i]) and np.isfinite(atr) and atr > 0:
                in_trade = True
                entry_price = close
                entry_atr = float(atr)
                held = 0
                signal.iloc[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal.shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
