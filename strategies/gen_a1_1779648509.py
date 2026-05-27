from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TrendSharpeAtrParams:
    sharpe_window: int = 20
    atr_window: int = 14
    atr_ratio_window: int = 50
    fast_dir_window: int = 5
    sharpe_enter: float = 0.10
    sharpe_exit: float = 0.03
    atr_ratio_max: float = 0.95
    breakeven_pct: float = 0.012
    trail_atr_mult: float = 2.5
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[TrendSharpeAtrParams]):
    strategy_id = "gen_a1_1779648509"

    @classmethod
    def params_type(cls):
        return TrendSharpeAtrParams

    @classmethod
    def warmup_bars(cls, params: TrendSharpeAtrParams) -> int:
        return int(
            max(
                params.sharpe_window,
                params.atr_ratio_window,
                params.atr_window,
                params.fast_dir_window,
            )
            + 2
        )

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: TrendSharpeAtrParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        atr_long = tr.rolling(params.atr_ratio_window, min_periods=params.atr_ratio_window).mean()
        atr_ratio = atr / atr_long.replace(0, np.nan)

        rets = close.pct_change()
        mean_ret = rets.rolling(params.sharpe_window, min_periods=params.sharpe_window).mean()
        std_ret = rets.rolling(params.sharpe_window, min_periods=params.sharpe_window).std()
        sharpe = mean_ret / std_ret.replace(0, np.nan)

        fast_dir = rets.rolling(params.fast_dir_window, min_periods=params.fast_dir_window).mean()

        out = pd.DataFrame(
            {
                "atr": atr,
                "atr_ratio": atr_ratio,
                "sharpe": sharpe,
                "fast_dir": fast_dir,
            },
            index=data.index,
        )
        return out

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TrendSharpeAtrParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy()
        high = data["high"].to_numpy()
        low = data["low"].to_numpy()
        atr = indicators["atr"].to_numpy()
        atr_ratio = indicators["atr_ratio"].to_numpy()
        sharpe = indicators["sharpe"].to_numpy()
        fast_dir = indicators["fast_dir"].to_numpy()

        n = len(close)
        raw_signal = np.zeros(n, dtype=np.int64)

        regime_long = False
        regime_short = False

        position = 0
        entry_price = 0.0
        entry_bar = -1
        stop_price = 0.0
        breakeven_armed = False
        peak_price = 0.0
        trough_price = 0.0

        vol_expand_thresh = 1.0 + (1.0 - params.atr_ratio_max)

        for i in range(n):
            s = sharpe[i]
            ar = atr_ratio[i]
            at = atr[i]
            fd = fast_dir[i]

            # Hysteresis regime tracking: both volatility primitives must agree to arm
            if (not np.isnan(s)) and (not np.isnan(ar)):
                vol_contracted = ar < params.atr_ratio_max
                if vol_contracted:
                    if s >= params.sharpe_enter:
                        regime_long = True
                        regime_short = False
                    elif s <= -params.sharpe_enter:
                        regime_short = True
                        regime_long = False
                if regime_long and s < params.sharpe_exit:
                    regime_long = False
                if regime_short and s > -params.sharpe_exit:
                    regime_short = False
                if ar > vol_expand_thresh:
                    regime_long = False
                    regime_short = False

            # Manage an open position: breakeven-then-ATR-trail exit
            if position != 0:
                bars_held = i - entry_bar
                if position == 1:
                    if high[i] > peak_price:
                        peak_price = high[i]
                else:
                    if low[i] < trough_price:
                        trough_price = low[i]

                if position == 1 and not breakeven_armed:
                    if entry_price > 0 and (peak_price - entry_price) / entry_price >= params.breakeven_pct:
                        if entry_price > stop_price:
                            stop_price = entry_price
                        breakeven_armed = True
                elif position == -1 and not breakeven_armed:
                    if entry_price > 0 and (entry_price - trough_price) / entry_price >= params.breakeven_pct:
                        if entry_price < stop_price:
                            stop_price = entry_price
                        breakeven_armed = True

                if breakeven_armed and not np.isnan(at):
                    if position == 1:
                        cand = peak_price - params.trail_atr_mult * at
                        if cand > stop_price:
                            stop_price = cand
                    else:
                        cand = trough_price + params.trail_atr_mult * at
                        if cand < stop_price:
                            stop_price = cand

                exit_now = False
                if position == 1 and low[i] <= stop_price:
                    exit_now = True
                elif position == -1 and high[i] >= stop_price:
                    exit_now = True
                elif bars_held >= params.max_hold_bars:
                    exit_now = True

                if exit_now:
                    raw_signal[i] = 0
                    position = 0
                    entry_price = 0.0
                    entry_bar = -1
                    stop_price = 0.0
                    breakeven_armed = False
                    peak_price = 0.0
                    trough_price = 0.0
                else:
                    raw_signal[i] = position
                continue

            # No position - consider new entry with regime gate + fast directional confirmation
            if np.isnan(at) or np.isnan(fd) or np.isnan(s) or np.isnan(ar):
                raw_signal[i] = 0
                continue

            if regime_long and fd > 0.0:
                position = 1
                entry_price = close[i]
                entry_bar = i
                stop_price = close[i] - params.trail_atr_mult * at
                breakeven_armed = False
                peak_price = high[i]
                trough_price = low[i]
                raw_signal[i] = 1
            elif regime_short and fd < 0.0:
                position = -1
                entry_price = close[i]
                entry_bar = i
                stop_price = close[i] + params.trail_atr_mult * at
                breakeven_armed = False
                peak_price = high[i]
                trough_price = low[i]
                raw_signal[i] = -1
            else:
                raw_signal[i] = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
