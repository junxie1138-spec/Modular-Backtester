from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_period: int = 200
    gap_lookback: int = 60
    gap_threshold_pct: float = 0.30
    autocorr_threshold: float = 0.15
    atr_period: int = 14
    atr_stop_mult: float = 2.5
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779652259"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(max(params.ma_period, params.gap_lookback + 2, params.atr_period + 1)) + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)

        # Overnight gap as a return (open vs prior close)
        gap = (open_ / prev_close) - 1.0

        # 200-day MA regime polarity filter
        ma_long = close.rolling(params.ma_period, min_periods=params.ma_period).mean()

        # Rolling lag-1 autocorrelation of the gap series.
        # This is the "tide phase": positive => flood (gap-momentum),
        # negative => ebb (gap-reversion).
        gap_lag = gap.shift(1)
        win = params.gap_lookback
        g_mean = gap.rolling(win, min_periods=win).mean()
        l_mean = gap_lag.rolling(win, min_periods=win).mean()
        g_dev = gap - g_mean
        l_dev = gap_lag - l_mean
        cov = (g_dev * l_dev).rolling(win, min_periods=win).mean()
        var_g = (g_dev * g_dev).rolling(win, min_periods=win).mean()
        var_l = (l_dev * l_dev).rolling(win, min_periods=win).mean()
        denom = np.sqrt(var_g * var_l)
        denom = denom.replace(0.0, np.nan)
        gap_autocorr = cov / denom

        # ATR for fixed volatility-stop (simple SMA of true range)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        return pd.DataFrame(
            {
                "gap": gap,
                "ma_long": ma_long,
                "gap_autocorr": gap_autocorr,
                "atr": atr,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        gap = indicators["gap"].to_numpy(dtype=float)
        ma_long = indicators["ma_long"].to_numpy(dtype=float)
        ac = indicators["gap_autocorr"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        size = np.ones(n, dtype=np.float64)

        gap_thresh = params.gap_threshold_pct / 100.0
        ac_thresh = float(params.autocorr_threshold)
        atr_mult = float(params.atr_stop_mult)
        max_hold = int(params.max_hold_bars)

        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            c = close[i]

            ind_ready = (
                not np.isnan(ma_long[i])
                and not np.isnan(ac[i])
                and not np.isnan(atr[i])
                and not np.isnan(gap[i])
                and not np.isnan(c)
            )

            # --- Exit logic: fixed vol-stop relative to entry price ---
            if position == 1:
                stop = entry_price - atr_mult * entry_atr
                bars_held += 1
                if (not np.isnan(c)) and (c < stop or bars_held >= max_hold):
                    position = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0
            elif position == -1:
                stop = entry_price + atr_mult * entry_atr
                bars_held += 1
                if (not np.isnan(c)) and (c > stop or bars_held >= max_hold):
                    position = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0

            # --- Entry logic: only when flat and indicators are ready ---
            if position == 0 and ind_ready:
                above = c > ma_long[i]
                persistence = ac[i] > ac_thresh
                reversion = ac[i] < -ac_thresh
                up_gap = gap[i] > gap_thresh
                down_gap = gap[i] < -gap_thresh

                go_long = False
                go_short = False

                if above:
                    # Bullish polarity: only take longs
                    if persistence and up_gap:
                        go_long = True  # ride the flood-tide up-gap
                    elif reversion and down_gap:
                        go_long = True  # fade ebb-tide down-gap
                else:
                    # Bearish polarity: only take shorts
                    if persistence and down_gap:
                        go_short = True  # ride the flood-tide down-gap
                    elif reversion and up_gap:
                        go_short = True  # fade ebb-tide up-gap

                if go_long and (not np.isnan(atr[i])) and atr[i] > 0.0:
                    position = 1
                    entry_price = c
                    entry_atr = atr[i]
                    bars_held = 0
                elif go_short and (not np.isnan(atr[i])) and atr[i] > 0.0:
                    position = -1
                    entry_price = c
                    entry_atr = atr[i]
                    bars_held = 0

            raw_signal[i] = position

        df = pd.DataFrame(
            {
                "signal": raw_signal,
                "size": size,
            },
            index=data.index,
        )

        # MANDATORY one-bar shift: decision on bar N close, fill on bar N+1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
