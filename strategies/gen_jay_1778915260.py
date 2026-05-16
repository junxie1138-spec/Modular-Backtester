from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SellEpidemicExhaustionParams:
    epidemic_window: int = 12
    epidemic_peak_lag: int = 4
    vol_ma_period: int = 20
    price_ma_period: int = 15
    regime_ma: int = 200
    atr_period: int = 14
    breakeven_pct: float = 1.5
    trail_atr_mult: float = 2.0
    min_infected: int = 3


class GeneratedStrategy(BaseStrategy[SellEpidemicExhaustionParams]):
    strategy_id = "gen_jay_1778915260"

    @classmethod
    def params_type(cls) -> type[SellEpidemicExhaustionParams]:
        return SellEpidemicExhaustionParams

    @staticmethod
    def warmup_bars(params: SellEpidemicExhaustionParams) -> int:
        return params.regime_ma + params.epidemic_window + params.epidemic_peak_lag + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: SellEpidemicExhaustionParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        prev_close = close.shift(1)

        ind["regime_ma"] = close.rolling(params.regime_ma).mean()
        ind["bull"] = (close > ind["regime_ma"]).astype(float)

        ind["vol_ma"] = volume.rolling(params.vol_ma_period).mean()

        down_mask = (close < prev_close).astype(float)
        high_vol_mask = (volume > ind["vol_ma"]).astype(float)
        ind["infected"] = down_mask * high_vol_mask

        ind["infected_count"] = ind["infected"].rolling(params.epidemic_window).sum()
        ind["infected_past"] = ind["infected_count"].shift(params.epidemic_peak_lag)
        ind["epidemic_waning"] = (
            (ind["infected_count"] < ind["infected_past"]) &
            (ind["infected_past"] >= params.min_infected)
        ).astype(float)

        down_vol = volume.where(close < prev_close, 0.0)
        vol_ma_safe = ind["vol_ma"].replace(0.0, np.nan)
        ind["down_vol_ratio"] = (
            (down_vol / vol_ma_safe).rolling(params.epidemic_window).mean()
        )
        ind["down_vol_depleting"] = (
            ind["down_vol_ratio"] < ind["down_vol_ratio"].shift(params.epidemic_peak_lag)
        ).astype(float)

        ind["price_ma"] = close.rolling(params.price_ma_period).mean()
        ind["below_price_ma"] = (close < ind["price_ma"]).astype(float)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SellEpidemicExhaustionParams,
    ) -> SignalFrame:
        close = data["close"].values
        low = data["low"].values
        bull = indicators["bull"].fillna(0.0).values
        epidemic_waning = indicators["epidemic_waning"].fillna(0.0).values
        down_vol_depleting = indicators["down_vol_depleting"].fillna(0.0).values
        below_price_ma = indicators["below_price_ma"].fillna(0.0).values
        atr = indicators["atr"].fillna(0.0).values

        n = len(close)

        raw_entry = (
            (bull == 1.0) &
            (epidemic_waning == 1.0) &
            (down_vol_depleting == 1.0) &
            (below_price_ma == 1.0)
        )

        final_signal = np.zeros(n, dtype=int)
        final_size = np.ones(n, dtype=float)

        in_trade = False
        entry_price = 0.0
        stop_price = 0.0
        breakeven_triggered = False

        for i in range(n):
            if in_trade:
                atr_i = atr[i] if atr[i] > 0.0 else 1e-8
                if low[i] <= stop_price:
                    final_signal[i] = 0
                    in_trade = False
                    breakeven_triggered = False
                else:
                    final_signal[i] = 1
                    final_size[i] = 1.0
                    if not breakeven_triggered:
                        if close[i] >= entry_price * (1.0 + params.breakeven_pct / 100.0):
                            stop_price = max(stop_price, entry_price)
                            breakeven_triggered = True
                    trail_level = close[i] - params.trail_atr_mult * atr_i
                    if trail_level > stop_price:
                        stop_price = trail_level
            else:
                if i > 0 and raw_entry[i - 1]:
                    atr_i = atr[i] if atr[i] > 0.0 else 1e-8
                    entry_price = close[i]
                    stop_price = entry_price - params.trail_atr_mult * atr_i
                    breakeven_triggered = False
                    in_trade = True
                    final_signal[i] = 1
                    final_size[i] = 1.0
                else:
                    final_signal[i] = 0

        df = data[["close"]].copy()
        df["signal"] = final_signal
        df["size"] = final_size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
