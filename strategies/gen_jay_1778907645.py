from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EpidemicVolRegimeParams:
    atr_period: int = 14
    infection_window: int = 20
    vol_threshold_pct: float = 1.2
    trend_period: int = 10
    g_percentile: float = 65.0
    breakeven_pct: float = 1.5
    trail_atr_mult: float = 1.5
    initial_stop_mult: float = 2.0


class GeneratedStrategy(BaseStrategy[EpidemicVolRegimeParams]):
    strategy_id = "gen_jay_1778907645"

    @classmethod
    def params_type(cls):
        return EpidemicVolRegimeParams

    @staticmethod
    def warmup_bars(params: EpidemicVolRegimeParams) -> int:
        return max(params.atr_period, params.infection_window, params.trend_period) + 252 + 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: EpidemicVolRegimeParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()
        ind["atr"] = atr

        vol_ratio_pct = (atr / close) * 100.0
        ind["bar_infected"] = (vol_ratio_pct > params.vol_threshold_pct).astype(float)

        I_t = ind["bar_infected"].rolling(params.infection_window).mean()
        S_t = 1.0 - I_t
        ind["I_t"] = I_t

        G_t = S_t * I_t
        ind["G_t"] = G_t
        ind["dG_dt"] = G_t.diff()
        ind["G_pct"] = G_t.rolling(252).rank(pct=True) * 100.0

        ema = close.ewm(span=params.trend_period, adjust=False).mean()
        ema_prev = ema.shift(params.trend_period // 2 + 1)
        ind["trend_sign"] = np.sign(ema - ema_prev)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EpidemicVolRegimeParams,
    ) -> SignalFrame:
        close = data["close"].values
        low_arr = data["low"].values
        high_arr = data["high"].values
        atr = indicators["atr"].values
        I_t = indicators["I_t"].values
        G_t = indicators["G_t"].values
        dG_dt = indicators["dG_dt"].values
        G_pct = indicators["G_pct"].values
        trend_sign = indicators["trend_sign"].values

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.95)

        in_trade = False
        entry_price = 0.0
        direction = 0
        stop_price = 0.0
        breakeven_hit = False
        breakeven_mult = 1.0 + params.breakeven_pct / 100.0

        for i in range(n):
            c = close[i]
            a = atr[i]

            if np.isnan(a) or np.isnan(G_pct[i]) or np.isnan(I_t[i]) or np.isnan(dG_dt[i]):
                signal[i] = 0
                size[i] = 0.95
                continue

            g_scaled = min(G_t[i] * 2.0, 0.45) if not np.isnan(G_t[i]) else 0.0
            trade_size = 0.5 + g_scaled

            if in_trade:
                if direction == 1:
                    if low_arr[i] <= stop_price:
                        signal[i] = 0
                        in_trade = False
                        direction = 0
                    else:
                        if not breakeven_hit and c >= entry_price * breakeven_mult:
                            stop_price = entry_price
                            breakeven_hit = True
                        if breakeven_hit:
                            trail = c - params.trail_atr_mult * a
                            if trail > stop_price:
                                stop_price = trail
                        signal[i] = 1
                        size[i] = trade_size
                elif direction == -1:
                    if high_arr[i] >= stop_price:
                        signal[i] = 0
                        in_trade = False
                        direction = 0
                    else:
                        if not breakeven_hit and c <= entry_price / breakeven_mult:
                            stop_price = entry_price
                            breakeven_hit = True
                        if breakeven_hit:
                            trail = c + params.trail_atr_mult * a
                            if trail < stop_price:
                                stop_price = trail
                        signal[i] = -1
                        size[i] = trade_size
            else:
                it = I_t[i]
                dg = dG_dt[i]
                g = G_pct[i]
                ts = trend_sign[i]

                early = (it < 0.5) and (dg > 0.0) and (g >= params.g_percentile)
                late = (it > 0.5) and (dg < 0.0) and (g >= params.g_percentile)

                if early and ts > 0:
                    signal[i] = 1
                    in_trade = True
                    direction = 1
                    entry_price = c
                    stop_price = c - params.initial_stop_mult * a
                    breakeven_hit = False
                    size[i] = trade_size
                elif early and ts < 0:
                    signal[i] = -1
                    in_trade = True
                    direction = -1
                    entry_price = c
                    stop_price = c + params.initial_stop_mult * a
                    breakeven_hit = False
                    size[i] = trade_size
                elif late and ts < 0:
                    signal[i] = 1
                    in_trade = True
                    direction = 1
                    entry_price = c
                    stop_price = c - params.initial_stop_mult * a
                    breakeven_hit = False
                    size[i] = trade_size
                elif late and ts > 0:
                    signal[i] = -1
                    in_trade = True
                    direction = -1
                    entry_price = c
                    stop_price = c + params.initial_stop_mult * a
                    breakeven_hit = False
                    size[i] = trade_size

        df = data[["close"]].copy()
        df["signal"] = signal
        df["size"] = size

        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
