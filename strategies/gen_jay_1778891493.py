from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    drawdown_window: int = 60
    drawdown_threshold: float = 0.05
    vol_window: int = 10
    buyer_dom_threshold: float = 0.55
    vol_avg_window: int = 20
    vol_multiplier: float = 1.1
    profit_target: float = 0.06
    time_stop: int = 21


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778891493"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.drawdown_window, params.vol_avg_window) + params.vol_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        volume = data["volume"]

        rolling_high = close.rolling(params.drawdown_window).max()
        ind["drawdown_pct"] = (close - rolling_high) / rolling_high.replace(0, np.nan)

        price_up = (close > close.shift(1)).astype(float)
        up_volume = volume * price_up

        rolling_up = up_volume.rolling(params.vol_window).sum()
        rolling_total = volume.rolling(params.vol_window).sum()
        buyer_dom = rolling_up / rolling_total.replace(0, np.nan)
        ind["buyer_dom"] = buyer_dom
        ind["buyer_dom_prev"] = buyer_dom.shift(1)

        avg_vol = volume.rolling(params.vol_avg_window).mean()
        ind["up_vol_ratio"] = up_volume / avg_vol.replace(0, np.nan)

        in_dd = (ind["drawdown_pct"] < -params.drawdown_threshold).astype(float)
        ind["had_drawdown"] = in_dd.rolling(params.vol_window).max()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"]
        n = len(close)

        buyer_dom = indicators["buyer_dom"]
        buyer_dom_prev = indicators["buyer_dom_prev"]
        up_vol_ratio = indicators["up_vol_ratio"]
        had_drawdown = indicators["had_drawdown"]

        crossover = (
            (buyer_dom >= params.buyer_dom_threshold)
            & (buyer_dom_prev < params.buyer_dom_threshold)
        )
        vol_confirmed = up_vol_ratio >= params.vol_multiplier
        raw_entry = (crossover & vol_confirmed & (had_drawdown >= 1.0)).fillna(False)

        dom_range = max(1.0 - params.buyer_dom_threshold, 1e-9)
        size_raw = 0.5 + 0.45 * (
            (buyer_dom - params.buyer_dom_threshold) / dom_range
        ).clip(0.0, 1.0)
        size_raw = size_raw.fillna(0.5)

        signal = pd.Series(0, index=close.index, dtype=int)
        size = pd.Series(0.5, index=close.index, dtype=float)

        in_trade = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if in_trade:
                bars_held += 1
                ret = (float(close.iloc[i]) - entry_price) / entry_price
                if ret >= params.profit_target or bars_held >= params.time_stop:
                    in_trade = False
                else:
                    signal.iloc[i] = 1
                    size.iloc[i] = float(size_raw.iloc[i])
            else:
                if bool(raw_entry.iloc[i]):
                    in_trade = True
                    entry_price = float(close.iloc[i])
                    bars_held = 0
                    signal.iloc[i] = 1
                    size.iloc[i] = float(size_raw.iloc[i])

        signal = signal.shift(1).fillna(0).astype(int)
        size = size.shift(1).fillna(0.5)

        df = pd.DataFrame({"signal": signal, "size": size}, index=close.index)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
