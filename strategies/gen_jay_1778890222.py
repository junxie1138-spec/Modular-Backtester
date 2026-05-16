from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    atr_pct_period: int = 63
    atr_threshold: float = 0.30
    autocorr_period: int = 12
    autocorr_threshold: float = -0.10
    mid_period: int = 20
    profit_target: float = 0.025
    max_hold_bars: int = 5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778890222"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.atr_pct_period + params.atr_period + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()

        ind["atr_pct_rank"] = atr.rolling(
            params.atr_pct_period, min_periods=params.atr_pct_period
        ).rank(pct=True)

        log_ret = np.log(close / close.shift(1))
        lagged_ret = log_ret.shift(1)
        ind["return_autocorr"] = log_ret.rolling(
            params.autocorr_period, min_periods=params.autocorr_period
        ).corr(lagged_ret)

        rolling_high = high.rolling(params.mid_period, min_periods=params.mid_period).max()
        rolling_low = low.rolling(params.mid_period, min_periods=params.mid_period).min()
        ind["range_mid"] = (rolling_high + rolling_low) / 2

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        atr_rank = indicators["atr_pct_rank"].values
        autocorr = indicators["return_autocorr"].values
        range_mid = indicators["range_mid"].values

        n = len(data)
        raw_signal = np.zeros(n, dtype=np.int64)
        size_arr = np.ones(n, dtype=np.float64)

        in_position = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            if np.isnan(atr_rank[i]) or np.isnan(autocorr[i]) or np.isnan(range_mid[i]):
                raw_signal[i] = 0
                continue

            if in_position:
                bars_held += 1
                pnl_pct = (close[i] - entry_price) / entry_price
                if pnl_pct >= params.profit_target or bars_held >= params.max_hold_bars:
                    raw_signal[i] = 0
                    in_position = False
                    bars_held = 0
                    entry_price = 0.0
                else:
                    raw_signal[i] = 1
            else:
                # Primitive 1: ATR below threshold percentile (spring compressed)
                # Primitive 2: lag-1 return autocorr sufficiently negative (elastic tension)
                # Both must agree; directional filter: close above rolling range midpoint
                if (
                    atr_rank[i] < params.atr_threshold
                    and autocorr[i] < params.autocorr_threshold
                    and close[i] > range_mid[i]
                ):
                    raw_signal[i] = 1
                    in_position = True
                    entry_price = close[i]
                    bars_held = 0
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame(
            {"signal": raw_signal, "size": size_arr}, index=data.index
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
