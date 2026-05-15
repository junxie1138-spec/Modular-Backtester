from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    drawdown_window: int = 20
    volume_window: int = 20
    atr_window: int = 14
    min_spring_tension: float = 2.0
    volume_mult: float = 1.1
    breakeven_pct: float = 0.5
    trail_atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778887422"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(params.drawdown_window, params.volume_window, params.atr_window) + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_window).mean()

        rolling_high = close.rolling(params.drawdown_window).max()
        ind["spring_tension"] = (rolling_high - close) / ind["atr"].replace(0, np.nan)

        vol_median = data["volume"].rolling(params.volume_window).median()
        is_bull = (close > data["open"]).astype(int)
        is_high_vol = (data["volume"] > vol_median * params.volume_mult).astype(int)
        vol_confirmed = is_bull * is_high_vol
        vol_confirmed_prev = vol_confirmed.shift(1).fillna(0).astype(int)

        ind["two_bar_confirm"] = (
            (vol_confirmed == 1) & (vol_confirmed_prev == 1)
        ).astype(int)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr_vals = indicators["atr"].values
        spring_tension = indicators["spring_tension"].values
        two_bar_confirm = indicators["two_bar_confirm"].values

        n = len(close)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        in_trade = False
        entry_price = 0.0
        stop_price = 0.0
        peak_price = 0.0
        breakeven_reached = False

        for i in range(n):
            c = close[i]
            st = spring_tension[i]
            tb = two_bar_confirm[i]
            a_raw = atr_vals[i]
            a = float(a_raw) if (not np.isnan(a_raw) and a_raw > 0) else 0.0

            cur_signal = 0
            cur_size = 1.0

            if in_trade:
                if c > peak_price:
                    peak_price = c

                if not breakeven_reached and c >= entry_price * (1.0 + params.breakeven_pct / 100.0):
                    breakeven_reached = True
                    if entry_price > stop_price:
                        stop_price = entry_price

                if breakeven_reached and a > 0:
                    new_trail = peak_price - params.trail_atr_mult * a
                    if new_trail > stop_price:
                        stop_price = new_trail

                if c <= stop_price:
                    in_trade = False
                else:
                    cur_signal = 1
                    cur_size = 1.0

            if not in_trade:
                st_ok = (not np.isnan(st)) and (float(st) >= params.min_spring_tension)
                tb_ok = int(tb) == 1
                if st_ok and tb_ok:
                    in_trade = True
                    entry_price = c
                    stop_price = (c - params.trail_atr_mult * a) if a > 0 else c * 0.98
                    peak_price = c
                    breakeven_reached = False
                    cur_signal = 1
                    cur_size = min(float(st) / max(params.min_spring_tension, 1e-6), 3.0)

            signal[i] = cur_signal
            size[i] = cur_size

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0)
        df["size"] = df["size"].clip(lower=0.01)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
