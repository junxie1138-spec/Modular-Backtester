from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    short_sell_window: int = 10
    long_sell_window: int = 60
    pct_lookback: int = 252
    exhaustion_pct: float = 0.25
    vol_confirm_pct: float = 0.65
    trend_window: int = 200
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778911729"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.trend_window + params.long_sell_window + params.pct_lookback + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        open_ = data["open"]

        is_down = (close < open_).astype(float)
        is_up = (close >= open_).astype(float)

        down_vol = volume * is_down
        up_vol = volume * is_up

        short_sell = down_vol.rolling(
            params.short_sell_window, min_periods=params.short_sell_window
        ).mean()
        long_sell = down_vol.rolling(
            params.long_sell_window, min_periods=params.long_sell_window
        ).mean()

        # Ratio of recent to historical selling pressure; NaN when denominator is zero
        sell_ratio = short_sell / long_sell.where(long_sell > 1e-10, np.nan)

        sell_ratio_pct = sell_ratio.rolling(
            params.pct_lookback, min_periods=params.pct_lookback
        ).rank(pct=True)
        up_vol_pct = up_vol.rolling(
            params.pct_lookback, min_periods=params.pct_lookback
        ).rank(pct=True)

        trend_ma = close.rolling(
            params.trend_window, min_periods=params.trend_window
        ).mean()
        # NaN trend_ma during warmup produces False comparison -> 0.0
        bull_regime = (close > trend_ma).astype(float)

        ind = pd.DataFrame(index=data.index)
        ind["sell_ratio_pct"] = sell_ratio_pct
        ind["up_vol_pct"] = up_vol_pct
        ind["bull_regime"] = bull_regime
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.ones(n, dtype=float)

        sell_ratio_pct = indicators["sell_ratio_pct"].values
        up_vol_pct = indicators["up_vol_pct"].values
        bull_regime = indicators["bull_regime"].values

        # NaN comparisons in numpy return False, so warmup bars are NaN-safe
        entry_arr = (
            (sell_ratio_pct <= params.exhaustion_pct)
            & (up_vol_pct >= params.vol_confirm_pct)
            & (bull_regime == 1.0)
        )

        hold_until = -1

        for i in range(n):
            if i <= hold_until:
                signal[i] = 1
            elif entry_arr[i]:
                signal[i] = 1
                hold_until = i + params.hold_bars - 1

        # Mandatory 1-bar shift: decision on bar N, fill on bar N+1
        sig_series = (
            pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        sz_series = pd.Series(size, index=data.index)

        df = pd.DataFrame(
            {"signal": sig_series, "size": sz_series}, index=data.index
        )
        return SignalFrame(data=df, signal_column="signal", size_column="size")
