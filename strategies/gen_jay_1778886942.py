from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TidalNodeParams:
    range_window: int = 20
    node_threshold: float = 0.70
    min_node_bars: int = 3
    expansion_mult: float = 1.20
    close_qual_min: float = 0.55
    ma_period: int = 200
    atr_period: int = 14
    breakeven_pct: float = 0.008
    trail_k: float = 1.5


class GeneratedStrategy(BaseStrategy["TidalNodeParams"]):
    strategy_id = "gen_jay_1778886942"

    @classmethod
    def params_type(cls):
        return TidalNodeParams

    @staticmethod
    def warmup_bars(params: TidalNodeParams) -> int:
        return params.ma_period + params.range_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: TidalNodeParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        hl = data["high"] - data["low"]
        range_median = hl.rolling(params.range_window).median()
        ind["range_ratio"] = hl / range_median.replace(0.0, np.nan)

        ind["close_qual"] = (data["close"] - data["low"]) / hl.replace(0.0, np.nan)

        ind["ma200"] = data["close"].rolling(params.ma_period).mean()
        ind["above_ma"] = (data["close"] > ind["ma200"]).astype(int)

        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                hl,
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_period).mean()

        # NaN-safe: treat missing range_ratio as not compressed
        rr_filled = ind["range_ratio"].fillna(params.node_threshold + 1.0)
        compressed = rr_filled < params.node_threshold
        streak = np.zeros(len(ind), dtype=int)
        cnt = 0
        for k, v in enumerate(compressed):
            cnt = cnt + 1 if v else 0
            streak[k] = cnt
        ind["compression_streak"] = streak

        # NaN-safe: treat missing range_ratio as not expanding
        ind["is_expansion"] = (
            ind["range_ratio"].fillna(0.0) > params.expansion_mult
        ).astype(int)
        ind["prev_streak"] = ind["compression_streak"].shift(1).fillna(0.0)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TidalNodeParams,
    ) -> SignalFrame:
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size_arr = np.ones(n, dtype=float)

        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        close_qual = indicators["close_qual"].to_numpy(dtype=float)
        above_ma = indicators["above_ma"].to_numpy(dtype=int)
        is_expansion = indicators["is_expansion"].to_numpy(dtype=int)
        prev_streak = indicators["prev_streak"].to_numpy(dtype=float)

        in_position = False
        entry_price = 0.0
        stop_price = 0.0
        at_breakeven = False
        energy_ratio = 1.0

        for i in range(n):
            if not in_position:
                atr_i = atr[i]
                pstreak = prev_streak[i]
                cq = close_qual[i]
                if (
                    not np.isnan(atr_i)
                    and atr_i > 0.0
                    and not np.isnan(cq)
                    and is_expansion[i] == 1
                    and pstreak >= params.min_node_bars
                    and cq >= params.close_qual_min
                    and above_ma[i] == 1
                ):
                    energy_ratio = min(
                        float(pstreak) / float(params.min_node_bars), 2.0
                    )
                    signal[i] = 1
                    size_arr[i] = energy_ratio
                    in_position = True
                    entry_price = close[i]
                    stop_price = entry_price - params.trail_k * atr_i
                    at_breakeven = False
            else:
                curr = close[i]
                atr_i = atr[i]
                if np.isnan(atr_i) or atr_i <= 0.0:
                    atr_i = (
                        atr[i - 1]
                        if i > 0 and not np.isnan(atr[i - 1]) and atr[i - 1] > 0.0
                        else 0.001
                    )

                if not at_breakeven and curr >= entry_price * (
                    1.0 + params.breakeven_pct
                ):
                    stop_price = max(stop_price, entry_price)
                    at_breakeven = True

                trail = curr - params.trail_k * atr_i
                if trail > stop_price:
                    stop_price = trail

                if curr <= stop_price:
                    signal[i] = 0
                    in_position = False
                    at_breakeven = False
                else:
                    signal[i] = 1
                    size_arr[i] = energy_ratio

        df = pd.DataFrame({"signal": signal, "size": size_arr}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
