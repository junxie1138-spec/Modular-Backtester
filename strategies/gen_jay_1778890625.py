from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    compression_window: int = 60
    price_window: int = 20
    shockwave_lag: int = 5
    compression_thresh: float = 0.25
    position_thresh: float = 0.60
    breakeven_pct: float = 0.01
    trail_atr_mult: float = 2.0
    atr_window: int = 14
    max_hold_bars: int = 3


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778890625"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.compression_window + params.shockwave_lag + params.atr_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        range_ratio = (data["high"] - data["low"]) / data["close"]

        cw = params.compression_window
        ind["compression_rank"] = range_ratio.rolling(cw, min_periods=cw).apply(
            lambda x: np.searchsorted(np.sort(x[:-1]), x[-1]) / max(len(x) - 1, 1),
            raw=True,
        )

        ind["lagged_compression"] = ind["compression_rank"].shift(params.shockwave_lag)

        pw = params.price_window
        ind["price_position_rank"] = data["close"].rolling(pw, min_periods=pw).apply(
            lambda x: np.searchsorted(np.sort(x[:-1]), x[-1]) / max(len(x) - 1, 1),
            raw=True,
        )

        aw = params.atr_window
        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - prev_close).abs(),
                (data["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ind["atr"] = tr.rolling(aw, min_periods=aw).mean()

        depth = (params.compression_thresh - ind["lagged_compression"]).clip(lower=0)
        ind["signal_strength"] = (depth / params.compression_thresh) * ind["price_position_rank"]

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        n = len(data)
        closes = data["close"].to_numpy(dtype=np.float64)
        highs = data["high"].to_numpy(dtype=np.float64)
        lows = data["low"].to_numpy(dtype=np.float64)

        lagged_comp = indicators["lagged_compression"].to_numpy(dtype=np.float64)
        price_rank = indicators["price_position_rank"].to_numpy(dtype=np.float64)
        atr_vals = indicators["atr"].to_numpy(dtype=np.float64)
        sig_strength = indicators["signal_strength"].to_numpy(dtype=np.float64)

        raw_signal = np.zeros(n, dtype=np.int64)
        raw_size = np.full(n, 0.5, dtype=np.float64)

        in_trade = False
        entry_price = 0.0
        stop_price = 0.0
        breakeven_reached = False
        hold_count = 0
        entry_size = 0.5

        for i in range(n):
            any_nan = (
                np.isnan(lagged_comp[i])
                or np.isnan(price_rank[i])
                or np.isnan(atr_vals[i])
                or np.isnan(sig_strength[i])
            )

            if in_trade:
                if any_nan:
                    raw_signal[i] = 1
                    raw_size[i] = entry_size
                    continue

                hold_count += 1
                stopped = lows[i] <= stop_price
                max_held = hold_count >= params.max_hold_bars

                if stopped or max_held:
                    raw_signal[i] = 0
                    raw_size[i] = entry_size
                    in_trade = False
                    breakeven_reached = False
                    hold_count = 0
                else:
                    if not breakeven_reached and highs[i] >= entry_price * (1.0 + params.breakeven_pct):
                        breakeven_reached = True
                        stop_price = max(stop_price, entry_price)

                    if breakeven_reached:
                        new_stop = closes[i] - params.trail_atr_mult * atr_vals[i]
                        stop_price = max(stop_price, new_stop)

                    raw_signal[i] = 1
                    raw_size[i] = entry_size
            else:
                if any_nan:
                    raw_signal[i] = 0
                    raw_size[i] = 0.5
                    continue

                comp_ok = lagged_comp[i] < params.compression_thresh
                rank_ok = price_rank[i] > params.position_thresh
                up_bar = i > 0 and closes[i] > closes[i - 1]

                if comp_ok and rank_ok and up_bar:
                    ss = sig_strength[i]
                    if np.isnan(ss) or ss <= 0.0:
                        ss = 0.05
                    entry_size = float(np.clip(ss, 0.05, 1.0))

                    raw_signal[i] = 1
                    raw_size[i] = entry_size
                    in_trade = True
                    entry_price = closes[i]
                    stop_price = entry_price - params.trail_atr_mult * atr_vals[i]
                    breakeven_reached = False
                    hold_count = 0
                else:
                    raw_signal[i] = 0
                    raw_size[i] = 0.5

        df = pd.DataFrame(
            {"signal": raw_signal.astype(float), "size": raw_size}, index=data.index
        )
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.5).clip(lower=0.05)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
