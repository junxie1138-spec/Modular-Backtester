from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    compress_window: int = 50
    range_pct_thresh: float = 0.30
    body_pct_thresh: float = 0.35
    return_pct_thresh: float = 0.60
    spike_pct_thresh: float = 0.80
    refractory_bars: int = 5
    profit_target_pct: float = 0.015
    time_stop_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778889988"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.compress_window + params.refractory_bars + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        hl_range = data["high"] - data["low"]
        hl_safe = hl_range.replace(0.0, np.nan)
        body = (data["close"] - data["open"]).abs()
        body_ratio = body / hl_safe

        ind["range_rank"] = hl_range.rolling(params.compress_window).rank(pct=True)
        ind["body_rank"] = body_ratio.rolling(params.compress_window).rank(pct=True)

        ret1 = data["close"].pct_change(1)
        ind["ret_rank"] = ret1.rolling(params.compress_window).rank(pct=True)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close = data["close"].values
        range_rank = indicators["range_rank"].values
        body_rank = indicators["body_rank"].values
        ret_rank = indicators["ret_rank"].values

        n = len(close)
        raw_signal = np.zeros(n, dtype=int)

        in_position = False
        entry_price = 0.0
        bars_held = 0
        warmup = params.compress_window + params.refractory_bars + 2

        for i in range(warmup, n):
            if in_position:
                bars_held += 1
                gain = (close[i] - entry_price) / entry_price
                if gain >= params.profit_target_pct or bars_held >= params.time_stop_bars:
                    raw_signal[i] = 0
                    in_position = False
                else:
                    raw_signal[i] = 1
            else:
                rr = range_rank[i]
                br = body_rank[i]
                rer = ret_rank[i]

                if not (np.isfinite(rr) and np.isfinite(br) and np.isfinite(rer)):
                    continue

                lo = i - params.refractory_bars
                recent_rr = range_rank[lo:i]
                had_spike = bool(np.any(recent_rr >= params.spike_pct_thresh))

                if (
                    not had_spike
                    and rr < params.range_pct_thresh
                    and br < params.body_pct_thresh
                    and rer > params.return_pct_thresh
                ):
                    raw_signal[i] = 1
                    in_position = True
                    entry_price = close[i]
                    bars_held = 0

        df = pd.DataFrame(index=data.index)
        df["signal"] = (
            pd.Series(raw_signal, index=data.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
