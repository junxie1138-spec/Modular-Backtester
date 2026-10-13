from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownEchoParams:
    peak_window: int = 30
    echo_window: int = 12
    min_depth: float = 0.035
    min_echo: float = 0.18
    breakeven_pct: float = 0.025
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[DrawdownEchoParams]):
    strategy_id = "gen_local-w2_1791295734"

    @classmethod
    def params_type(cls) -> type[DrawdownEchoParams]:
        return DrawdownEchoParams

    @staticmethod
    def warmup_bars(params: DrawdownEchoParams) -> int:
        return max(params.peak_window + params.echo_window + 1, 15)

    @staticmethod
    def indicators(data: pd.DataFrame, params: DrawdownEchoParams) -> pd.DataFrame:
        close = data["close"]
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = (1.0 - close / peak).clip(lower=0).fillna(0.0)
        change = depth.diff().fillna(0.0)
        echo = change.rolling(params.echo_window, min_periods=params.echo_window).corr(change.shift(1))
        previous_close = close.shift(1).fillna(close)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        return pd.DataFrame({
            "depth": depth,
            "depth_change": change,
            "echo": echo.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "atr": true_range.rolling(14, min_periods=14).mean().fillna(0.0),
        }, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownEchoParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.2, dtype=float)
        position = pending = 0
        entry = stop = 0.0
        held = 0
        position_size = 0.2

        for i in range(len(data)):
            if pending != position:
                position = pending
                entry = float(data["open"].iloc[i]) if position else 0.0
                stop = 0.0
                held = 0

            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            decision = position

            if position:
                held += 1
                if stop and ((position == 1 and close <= stop) or (position == -1 and close >= stop)):
                    decision = 0
                elif held >= params.max_hold:
                    decision = 0
                elif atr > 0:
                    if position == 1 and close >= entry * (1.0 + params.breakeven_pct):
                        stop = max(stop, entry, close - params.trail_atr * atr)
                    elif position == -1 and close <= entry * (1.0 - params.breakeven_pct):
                        stop = min(stop, entry, close + params.trail_atr * atr) if stop else min(entry, close + params.trail_atr * atr)
            else:
                depth = float(indicators["depth"].iloc[i])
                change = float(indicators["depth_change"].iloc[i])
                echo = float(indicators["echo"].iloc[i])
                if depth >= params.min_depth and change > 0 and atr > 0:
                    if echo <= -params.min_echo:
                        decision = 1
                    elif echo >= params.min_echo:
                        decision = -1
                    if decision:
                        position_size = float(np.clip(
                            0.2 + 0.5 * min(depth / 0.10, 1.0)
                            + 0.25 * min((abs(echo) - params.min_echo) / 0.5, 1.0),
                            0.2, 0.95,
                        ))

            decisions[i] = decision
            sizes[i] = position_size if decision else 0.2
            pending = decision

        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.2).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
