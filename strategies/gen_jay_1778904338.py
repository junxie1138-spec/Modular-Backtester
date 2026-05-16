from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    gap_window: int = 40
    pct_window: int = 60
    infection_pct: float = 0.60
    pressure_pct: float = 0.72
    breakeven_pct: float = 1.2
    trail_atr_mult: float = 2.0


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778904338"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 2 * params.pct_window + params.gap_window + params.atr_period + 20

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        # ATR-normalised gap: open minus prior close scaled by recent volatility
        ind["gap_norm"] = (data["open"] - prev_close) / ind["atr"].replace(0, np.nan)
        abs_gap = ind["gap_norm"].abs()

        # Rolling percentile threshold (hard twist: adaptive, not fixed)
        ind["gap_thresh"] = abs_gap.rolling(
            params.pct_window, min_periods=params.pct_window
        ).quantile(params.infection_pct)

        # Infected bar: this gap exceeded the rolling percentile threshold
        ind["infected"] = (abs_gap >= ind["gap_thresh"]).astype(float)
        ind.loc[ind["gap_thresh"].isna(), "infected"] = np.nan

        # Directional infection: sign tracks whether the gap was up or down
        gap_sign = np.sign(ind["gap_norm"]).fillna(0.0)
        ind["dir_infected"] = ind["infected"] * gap_sign

        # Epidemic infection rate: fraction of infected bars in rolling window
        ind["infection_rate"] = ind["infected"].rolling(
            params.gap_window, min_periods=params.gap_window
        ).mean()

        # Net directional gap pressure: signed infections summed over window
        ind["net_pressure"] = ind["dir_infected"].rolling(
            params.gap_window, min_periods=params.gap_window
        ).sum()

        # Percentile rank of net pressure within its own rolling history
        ind["pressure_rank"] = ind["net_pressure"].rolling(
            params.pct_window, min_periods=params.pct_window
        ).rank(pct=True)

        # Immunity = susceptible fraction (1 - infection rate)
        ind["immunity"] = 1.0 - ind["infection_rate"]

        # Immunity slope: negative means epidemic is growing (R > 1 phase)
        ind["immunity_slope"] = ind["immunity"].diff(3)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)

        close_arr = data["close"].values
        atr_arr = indicators["atr"].values
        gap_norm_arr = indicators["gap_norm"].values
        pressure_rank_arr = indicators["pressure_rank"].values
        immunity_slope_arr = indicators["immunity_slope"].values

        # NaN comparisons in numpy return False, so NaN bars naturally suppress entry
        long_cond = (
            (pressure_rank_arr >= params.pressure_pct)
            & (immunity_slope_arr < 0.0)
            & (gap_norm_arr > 0.0)
        )
        short_cond = (
            (pressure_rank_arr <= (1.0 - params.pressure_pct))
            & (immunity_slope_arr < 0.0)
            & (gap_norm_arr < 0.0)
        )

        raw_signal = np.zeros(n, dtype=int)
        breakeven_mult = params.breakeven_pct / 100.0

        position = 0
        entry_price = 0.0
        stop_price = 0.0
        breakeven_triggered = False

        for i in range(n):
            cur_close = close_arr[i]
            cur_atr = atr_arr[i]

            if np.isnan(cur_close) or np.isnan(cur_atr):
                raw_signal[i] = position
                continue

            if position == 0:
                if bool(long_cond[i]):
                    position = 1
                    entry_price = cur_close
                    stop_price = cur_close - params.trail_atr_mult * cur_atr
                    breakeven_triggered = False
                    raw_signal[i] = 1
                elif bool(short_cond[i]):
                    position = -1
                    entry_price = cur_close
                    stop_price = cur_close + params.trail_atr_mult * cur_atr
                    breakeven_triggered = False
                    raw_signal[i] = -1
                else:
                    raw_signal[i] = 0

            elif position == 1:
                pnl_pct = (cur_close - entry_price) / entry_price
                if not breakeven_triggered and pnl_pct >= breakeven_mult:
                    breakeven_triggered = True
                    stop_price = max(stop_price, entry_price)
                if breakeven_triggered:
                    stop_price = max(stop_price, cur_close - params.trail_atr_mult * cur_atr)
                if cur_close <= stop_price:
                    position = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1

            else:  # position == -1
                pnl_pct = (entry_price - cur_close) / entry_price
                if not breakeven_triggered and pnl_pct >= breakeven_mult:
                    breakeven_triggered = True
                    stop_price = min(stop_price, entry_price)
                if breakeven_triggered:
                    stop_price = min(stop_price, cur_close + params.trail_atr_mult * cur_atr)
                if cur_close >= stop_price:
                    position = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = -1

        df = pd.DataFrame(
            {"signal": raw_signal, "size": np.ones(n, dtype=float)},
            index=data.index,
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
