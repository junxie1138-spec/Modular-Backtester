from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 50
    z_threshold: float = -1.5
    infection_window: int = 20
    min_infection: float = 0.55
    atr_window: int = 14
    atr_stop_k: float = 2.0
    hold_bars: int = 5


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778912305"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.ma_window, params.infection_window + 2, params.atr_window) + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        rolling_mean = data["close"].rolling(params.ma_window).mean()
        rolling_std = data["close"].rolling(params.ma_window).std()
        safe_std = rolling_std.where(rolling_std > 0, np.nan)
        ind["z_score"] = (data["close"] - rolling_mean) / safe_std

        returns = data["close"].pct_change()
        is_down = (returns < 0).astype(float)
        ind["infection_rate"] = is_down.rolling(params.infection_window).mean()
        ind["infection_delta"] = ind["infection_rate"].diff()

        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - data["close"].shift(1)).abs(),
                (data["low"] - data["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.95, dtype=float)

        closes = data["close"].values
        z_scores = indicators["z_score"].values
        infection_rates = indicators["infection_rate"].values
        infection_deltas = indicators["infection_delta"].values
        atrs = indicators["atr"].values

        in_trade = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(1, n):
            if in_trade:
                bars_held += 1
                stop_price = entry_price - params.atr_stop_k * entry_atr

                if (not np.isnan(closes[i])) and closes[i] < stop_price:
                    signals[i] = 0
                    in_trade = False
                    bars_held = 0
                elif bars_held >= params.hold_bars:
                    signals[i] = 0
                    in_trade = False
                    bars_held = 0
                else:
                    signals[i] = 1
            else:
                if (
                    np.isnan(z_scores[i])
                    or np.isnan(infection_rates[i])
                    or np.isnan(infection_deltas[i])
                    or np.isnan(atrs[i])
                ):
                    continue

                p1 = z_scores[i] < params.z_threshold
                p2 = (
                    infection_rates[i] >= params.min_infection
                    and infection_deltas[i] < 0.0
                )

                if p1 and p2:
                    signals[i] = 1
                    in_trade = True
                    entry_price = closes[i]
                    entry_atr = atrs[i]
                    bars_held = 0

        df = data[["close"]].copy()
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = sizes

        return SignalFrame(data=df, signal_column="signal", size_column="size")
