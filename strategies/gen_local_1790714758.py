from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    return_window: int = 20
    volume_window: int = 20
    volume_multiple: float = 1.5
    center_window: int = 5
    shock_multiple: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790714758"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    @classmethod
    def warmup_bars(cls, params: GeneratedParams) -> int:
        return max(params.return_window + 1, params.volume_window, params.center_window + 1)

    @classmethod
    def indicators(cls, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        ret = data["close"].pct_change()
        sigma = ret.rolling(params.return_window).std()
        volume_mean = data["volume"].rolling(params.volume_window).mean()
        center = data["close"].rolling(params.center_window).mean()
        return pd.DataFrame({
            "ret": ret,
            "sigma": sigma,
            "volume_mean": volume_mean,
            "center": center,
        }, index=data.index)

    @classmethod
    def generate_signals(
        cls,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        ret = indicators["ret"]
        sigma = indicators["sigma"]
        shock = ret.abs() > params.shock_multiple * sigma
        volume_confirm = data["volume"] > params.volume_multiple * indicators["volume_mean"]
        displaced = data["close"] - indicators["center"]
        restoring = displaced * displaced.shift(1) < 0
        raw = pd.Series(0, index=data.index, dtype=int)
        short = shock & volume_confirm & restoring & (ret > 0)
        long = shock & volume_confirm & restoring & (ret < 0)
        raw.loc[short.fillna(False)] = -1
        raw.loc[long.fillna(False)] = 1

        signal = raw.shift(1).fillna(0).astype(int)
        out = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        active = 0
        age = 0
        for i in range(len(out)):
            if active:
                age += 1
                if age >= params.hold_bars:
                    out.iloc[i, out.columns.get_loc("signal")] = 0
                    active = 0
                    age = 0
            elif out.iloc[i, out.columns.get_loc("signal")] != 0:
                active = 1
                age = 0
        return SignalFrame(data=out, signal_column="signal", size_column="size")
