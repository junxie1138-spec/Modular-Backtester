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
    gap_tension_z: float = 1.2
    season_start_month: int = 10
    season_end_month: int = 4
    hold_bars: int = 7
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_jay_1778882084"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.ma_period + params.gap_lookback + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        ind["ma"] = data["close"].rolling(params.ma_period).mean()
        prev_close = data["close"].shift(1)
        ind["gap"] = (data["open"] - prev_close) / prev_close
        ind["gap_mean"] = ind["gap"].rolling(params.gap_lookback).mean()
        ind["gap_std"] = ind["gap"].rolling(params.gap_lookback).std()
        ind["month"] = data.index.month
        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = data.copy()

        # Bull regime: close above 200-day MA
        regime = data["close"] > indicators["ma"]

        # Seasonal window: bullish calendar season (default Oct-Apr, wraps year boundary)
        month = indicators["month"]
        ssm = params.season_start_month
        sem = params.season_end_month
        if ssm > sem:
            seasonal = (month >= ssm) | (month <= sem)
        else:
            seasonal = (month >= ssm) & (month <= sem)

        # Spring tension: gap z-score shows an abnormally stretched down-gap
        gap_std_safe = indicators["gap_std"].replace(0, np.nan)
        gap_z = (indicators["gap"] - indicators["gap_mean"]) / gap_std_safe
        stretched = gap_z < -params.gap_tension_z

        # Combined entry: all three conditions met at bar close
        raw_entry = (regime & seasonal & stretched).fillna(False).to_numpy()

        # Fixed-bar exit loop: hold exactly hold_bars bars, no signal-based exit
        n = len(df)
        raw_signal = np.zeros(n, dtype=int)
        in_trade = False
        bars_held = 0

        for i in range(n):
            if in_trade:
                raw_signal[i] = 1
                bars_held += 1
                if bars_held >= params.hold_bars:
                    in_trade = False
                    bars_held = 0
            elif raw_entry[i]:
                raw_signal[i] = 1
                in_trade = True
                bars_held = 1

        # Shift by 1: decide on bar N close, fill on bar N+1 open
        df["signal"] = (
            pd.Series(raw_signal, index=df.index).shift(1).fillna(0).astype(int)
        )
        df["size"] = params.size

        return SignalFrame(data=df, signal_column="signal", size_column="size")
