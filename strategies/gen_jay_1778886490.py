from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    drawdown_thresh: float = 0.025
    profit_target_pct: float = 0.012


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778886490"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 21

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        roll = 20

        rolling_high = close.rolling(roll).max()
        drawdown_depth = (rolling_high - close) / rolling_high.replace(0, np.nan)

        daily_range = high - low
        avg_range = daily_range.rolling(roll).mean()
        range_ratio = daily_range / avg_range.replace(0, np.nan)

        ind = pd.DataFrame(index=data.index)
        ind["drawdown_depth"] = drawdown_depth
        ind["range_ratio"] = range_ratio
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        dd = indicators["drawdown_depth"].values
        rr = indicators["range_ratio"].values

        n = len(close)
        pre_signal = np.zeros(n, dtype=int)

        in_trade = False
        trade_dir = 0
        bars_held = 0
        entry_close = 0.0

        for i in range(n):
            if np.isnan(dd[i]) or np.isnan(rr[i]):
                continue

            if in_trade:
                bars_held += 1
                ret = trade_dir * (close[i] - entry_close) / max(entry_close, 1e-9)
                if ret >= params.profit_target_pct or bars_held >= 2:
                    in_trade = False
                    trade_dir = 0
                    # pre_signal[i] stays 0: exit bar
                else:
                    pre_signal[i] = trade_dir
            else:
                if dd[i] > params.drawdown_thresh and rr[i] < 0.85:
                    # Elastic: deep drawdown + compressed range -> bounce long
                    pre_signal[i] = 1
                    in_trade = True
                    trade_dir = 1
                    bars_held = 0
                    entry_close = close[i]
                elif dd[i] > params.drawdown_thresh and rr[i] > 1.15:
                    # Plastic: deep drawdown + expanding range -> continue short
                    pre_signal[i] = -1
                    in_trade = True
                    trade_dir = -1
                    bars_held = 0
                    entry_close = close[i]

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(pre_signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
