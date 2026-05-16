from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRankHysteresisParams:
    gap_window: int = 10
    rank_window: int = 63
    entry_pct: float = 20.0
    exit_pct: float = 55.0
    ma_period: int = 200


class GeneratedStrategy(BaseStrategy[GapRankHysteresisParams]):
    strategy_id = "gen_jay_1778913124"

    @classmethod
    def params_type(cls):
        return GapRankHysteresisParams

    @staticmethod
    def warmup_bars(params: GapRankHysteresisParams) -> int:
        return params.ma_period + params.rank_window + params.gap_window + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRankHysteresisParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)

        prev_close = data["close"].shift(1)
        overnight_gap = (data["open"] - prev_close) / prev_close

        cum_gap = overnight_gap.rolling(params.gap_window, min_periods=params.gap_window).sum()

        entry_q = params.entry_pct / 100.0
        exit_q = params.exit_pct / 100.0
        cum_gap_entry_level = (
            cum_gap.rolling(params.rank_window, min_periods=params.rank_window).quantile(entry_q)
        )
        cum_gap_exit_level = (
            cum_gap.rolling(params.rank_window, min_periods=params.rank_window).quantile(exit_q)
        )

        ma200 = data["close"].rolling(params.ma_period, min_periods=params.ma_period).mean()

        ind["cum_gap"] = cum_gap
        ind["entry_level"] = cum_gap_entry_level
        ind["exit_level"] = cum_gap_exit_level
        ind["ma200"] = ma200

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapRankHysteresisParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 1.0

        cum_gap = indicators["cum_gap"].values
        entry_level = indicators["entry_level"].values
        exit_level = indicators["exit_level"].values
        ma200 = indicators["ma200"].values
        close = data["close"].values

        raw_signal = np.zeros(len(df), dtype=int)
        in_trade = False

        for i in range(len(df)):
            if (
                np.isnan(cum_gap[i])
                or np.isnan(entry_level[i])
                or np.isnan(exit_level[i])
                or np.isnan(ma200[i])
            ):
                raw_signal[i] = 0
                in_trade = False
                continue

            bull_regime = close[i] > ma200[i]

            if not in_trade:
                if bull_regime and cum_gap[i] <= entry_level[i]:
                    raw_signal[i] = 1
                    in_trade = True
                else:
                    raw_signal[i] = 0
            else:
                if cum_gap[i] > exit_level[i] or not bull_regime:
                    raw_signal[i] = 0
                    in_trade = False
                else:
                    raw_signal[i] = 1

        df["signal"] = pd.Series(raw_signal, index=df.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
