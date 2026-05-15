from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_window: int = 20
    autocorr_window: int = 15
    z_thresh: float = 0.5
    autocorr_thresh: float = 0.3


class GeneratedStrategy(BaseStrategy["Params"]):
    strategy_id = "gen_jay_1778887776"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return params.ma_window + params.autocorr_window + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        ma = close.rolling(params.ma_window).mean()
        std = close.rolling(params.ma_window).std()
        z = (close - ma) / std.where(std > 0)
        # Rolling lag-1 autocorrelation of the z-score: high value => displacement is
        # self-sustaining ("standing wave"), low/negative => mean-reverting noise
        z_autocorr = z.rolling(params.autocorr_window).corr(z.shift(1))
        return pd.DataFrame({"z": z, "z_autocorr": z_autocorr}, index=data.index)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        z = indicators["z"]
        z_autocorr = indicators["z_autocorr"]

        # Symmetric entry/exit: identical thresholds gate both legs.
        # Long when price is displaced positively from MA (z > z_thresh) AND
        # that displacement is autocorrelated above the autocorr_thresh, meaning
        # the z-score has been persistently positive (standing-wave antinode).
        # Exit fires on signal-reversal: when either condition fails the signal
        # drops to 0 with no separate stop logic needed.
        raw = (
            (z > params.z_thresh) & (z_autocorr > params.autocorr_thresh)
        ).astype(int)

        df = pd.DataFrame(index=data.index)
        # Mandatory one-bar shift: decision on bar N fills on bar N+1.
        df["signal"] = raw.shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
