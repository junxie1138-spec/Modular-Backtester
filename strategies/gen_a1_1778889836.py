from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalRankParams:
    ma_window: int = 200
    atr_window: int = 14
    horizon: int = 4
    rank_window: int = 252
    max_tdom: int = 22
    entry_hi: float = 0.75
    entry_lo: float = 0.45
    breakeven_pct: float = 0.015
    k_init: float = 2.0
    k_trail: float = 2.5
    max_hold: int = 5
    regime_exit: bool = True


class GeneratedStrategy(BaseStrategy[SeasonalRankParams]):
    strategy_id = "gen_a1_1778889836"

    @classmethod
    def params_type(cls):
        return SeasonalRankParams

    @staticmethod
    def warmup_bars(params: SeasonalRankParams) -> int:
        return max(params.ma_window, params.rank_window) + params.horizon + params.atr_window + 90

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalRankParams) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        out = pd.DataFrame(index=data.index)

        # 200-day regime MA (the hard twist).
        out["ma200"] = close.rolling(params.ma_window, min_periods=params.ma_window).mean()

        # ATR for the breakeven-then-trail exit.
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        # Trading-day-of-month calendar position (1..max_tdom).
        ym = data.index.to_period("M")
        tdom = data.groupby(ym).cumcount() + 1
        tdom = tdom.clip(upper=params.max_tdom)
        out["tdom"] = tdom.astype(float)

        # Realized forward return of each bar; only known `horizon` bars later.
        fwd_ret = close.shift(-params.horizon) / close - 1.0

        # Seasonal score: trailing mean forward-return for this calendar
        # position, using only occurrences whose outcome is already known.
        # expanding().mean().shift(1) inside each tdom group guarantees the
        # current bar's own (still-unknown) forward return is never used, and
        # the prior same-tdom occurrence sits ~21 bars back so its outcome is
        # already realized for any horizon < 21.
        seasonal_score = fwd_ret.groupby(tdom.values).transform(
            lambda s: s.expanding().mean().shift(1)
        )
        out["seasonal_score"] = seasonal_score

        # Rolling percentile rank of the current calendar position's seasonal
        # score against the trailing year of seasonal scores.
        out["seasonal_rank"] = seasonal_score.rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)

        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SeasonalRankParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy(dtype=float)
        ma200 = indicators["ma200"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        rank = indicators["seasonal_rank"].to_numpy(dtype=float)

        signal = np.zeros(n, dtype=int)

        pos = 0
        entry_price = 0.0
        stop = 0.0
        peak = 0.0
        armed = False
        bars_held = 0
        season_on = False  # hysteresis state for the seasonal regime

        for i in range(n):
            r = rank[i]
            if not np.isnan(r):
                if season_on:
                    if r < params.entry_lo:
                        season_on = False
                else:
                    if r > params.entry_hi:
                        season_on = True

            m = ma200[i]
            a = atr[i]
            regime_ok = (not np.isnan(m)) and close[i] > m

            if pos == 0:
                if season_on and regime_ok and (not np.isnan(a)) and a > 0.0:
                    pos = 1
                    entry_price = close[i]
                    peak = close[i]
                    armed = False
                    bars_held = 0
                    stop = entry_price - params.k_init * a
                    signal[i] = 1
            else:
                bars_held += 1
                if close[i] > peak:
                    peak = close[i]

                # Breakeven: once price has reached +breakeven_pct, lift the
                # stop to the entry price (never below it).
                if (not armed) and close[i] >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                    if entry_price > stop:
                        stop = entry_price

                # Trail: after breakeven, ratchet the stop up by k_trail*ATR
                # below the running peak. The stop only ever moves up.
                if armed and (not np.isnan(a)):
                    trail = peak - params.k_trail * a
                    if trail > stop:
                        stop = trail

                exit_now = False
                if close[i] <= stop:
                    exit_now = True
                if bars_held >= params.max_hold:
                    exit_now = True
                if params.regime_exit and (not np.isnan(m)) and close[i] < m:
                    exit_now = True

                if exit_now:
                    pos = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        # Mandatory one-bar shift: decide on bar N close, fill on bar N+1.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
