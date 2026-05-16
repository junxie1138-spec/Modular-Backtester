from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    er_window: int = 20
    rank_window: int = 60
    er_pct_threshold: float = 0.70
    dir_pct_threshold: float = 0.55
    profit_target_pct: float = 0.05
    time_stop_bars: int = 20


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778891863"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.rank_window + params.er_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]

        daily_abs = close.diff().abs()
        net_move_signed = close.diff(params.er_window)
        net_move_abs = net_move_signed.abs()
        path_length = daily_abs.rolling(params.er_window).sum()

        # Efficiency Ratio: fraction of total path that is directional (0=choppy, 1=clean trend)
        er = (net_move_abs / path_length.replace(0, np.nan)).clip(0, 1)

        # Adaptive percentile thresholds: rank ER and signed return over rolling history
        er_rank = er.rolling(params.rank_window).rank(pct=True)
        dir_rank = net_move_signed.rolling(params.rank_window).rank(pct=True)

        ind = pd.DataFrame(index=data.index)
        ind["er"] = er
        ind["er_rank"] = er_rank
        ind["dir_rank"] = dir_rank
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        close_arr = data["close"].values
        er_rank_arr = indicators["er_rank"].values
        dir_rank_arr = indicators["dir_rank"].values
        n = len(close_arr)

        raw_signal = np.zeros(n, dtype=np.int32)
        raw_size = np.ones(n, dtype=np.float64)

        in_position = False
        entry_price = 0.0
        bars_held = 0

        for i in range(n):
            er_v = er_rank_arr[i]
            dir_v = dir_rank_arr[i]

            if in_position:
                bars_held += 1
                gain = (close_arr[i] - entry_price) / entry_price
                if gain >= params.profit_target_pct or bars_held >= params.time_stop_bars:
                    in_position = False
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
                    raw_size[i] = er_v if not np.isnan(er_v) else 1.0
            else:
                if (
                    not np.isnan(er_v)
                    and not np.isnan(dir_v)
                    and er_v >= params.er_pct_threshold
                    and dir_v >= params.dir_pct_threshold
                ):
                    in_position = True
                    entry_price = close_arr[i]
                    bars_held = 0
                    raw_signal[i] = 1
                    raw_size[i] = er_v

        idx = data.index
        signal_s = pd.Series(raw_signal, index=idx, dtype=int)
        size_s = pd.Series(raw_size, index=idx)

        signal_s = signal_s.shift(1).fillna(0).astype(int)
        size_s = size_s.shift(1).fillna(1.0)

        df = pd.DataFrame({"signal": signal_s, "size": size_s}, index=idx)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
