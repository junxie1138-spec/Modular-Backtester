from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapQueueParams:
    atr_period: int = 14
    queue_period: int = 20
    capacity_threshold: float = 2.0
    ma_period: int = 200
    profit_target_pct: float = 4.0
    time_stop_bars: int = 18
    position_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapQueueParams]):
    strategy_id = "gen_jay_1778898743"

    @classmethod
    def params_type(cls):
        return GapQueueParams

    @staticmethod
    def warmup_bars(params: GapQueueParams) -> int:
        return params.ma_period + params.queue_period + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapQueueParams) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_period).mean()

        gap_norm = (open_ - prev_close) / atr.replace(0, np.nan)
        gap_pos = gap_norm.clip(lower=0)
        gap_load = gap_pos.rolling(params.queue_period).sum()

        ma200 = close.rolling(params.ma_period).mean()
        above_ma = (close > ma200).astype(int)

        ind = pd.DataFrame(index=data.index)
        ind["atr"] = atr
        ind["gap_norm"] = gap_norm
        ind["gap_load"] = gap_load
        ind["above_ma"] = above_ma
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapQueueParams,
    ) -> SignalFrame:
        close = data["close"].values
        gap_load = indicators["gap_load"].values
        gap_norm = indicators["gap_norm"].values
        above_ma = indicators["above_ma"].values
        n = len(close)

        raw_signal = np.zeros(n, dtype=int)
        in_position = False
        entry_close = 0.0
        bars_held = 0

        for i in range(n):
            if in_position:
                bars_held += 1
                ret = (close[i] - entry_close) / entry_close
                if (
                    ret >= params.profit_target_pct / 100.0
                    or bars_held >= params.time_stop_bars
                ):
                    in_position = False
                else:
                    raw_signal[i] = 1

            if not in_position:
                gl = gap_load[i]
                gn = gap_norm[i]
                if (
                    not np.isnan(gl)
                    and gl > params.capacity_threshold
                    and not np.isnan(gn)
                    and gn < 0.0
                    and above_ma[i] == 1
                ):
                    in_position = True
                    entry_close = close[i]
                    bars_held = 0
                    raw_signal[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw_signal
        df["size"] = params.position_size
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
