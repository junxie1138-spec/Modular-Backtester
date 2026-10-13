from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AnalogParams:
    lookback: int = 126
    min_samples: int = 10
    win_rate: float = 0.62
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AnalogParams]):
    strategy_id = "gen_local_1791226122"

    @classmethod
    def params_type(cls):
        return AnalogParams

    def warmup_bars(self, params: AnalogParams) -> int:
        return params.lookback + 8

    def indicators(self, data: pd.DataFrame, params: AnalogParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        accel = data["close"].pct_change().diff().replace([np.inf, -np.inf], np.nan)
        valid = accel.notna() & accel.shift(1).notna() & accel.shift(2).notna()
        shape = ((accel > 0).astype(int)
                 + 2 * (accel.shift(1) > 0).astype(int)
                 + 4 * (accel.shift(2) > 0).astype(int)).where(valid, -1).astype(int)
        out["accel"] = accel.fillna(0.0)
        out["win_rate"] = 0.0
        out["samples"] = 0.0
        completed_gain = data["close"].gt(data["close"].shift(4))
        for code in range(8):
            historical_shape = shape.shift(4).eq(code)
            samples = historical_shape.rolling(params.lookback, min_periods=params.lookback).sum()
            wins = (historical_shape & completed_gain).rolling(params.lookback, min_periods=params.lookback).sum()
            current_shape = shape.eq(code)
            out.loc[current_shape, "samples"] = samples[current_shape].fillna(0.0)
            out.loc[current_shape, "win_rate"] = (wins / samples.replace(0, np.nan))[current_shape].fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AnalogParams,
    ) -> SignalFrame:
        qualified = ((indicators["accel"] > 0)
                     & (indicators["samples"] >= params.min_samples)
                     & (indicators["win_rate"] >= params.win_rate))
        confirmed = (qualified & qualified.shift(1).fillna(False)).to_numpy(dtype=bool)
        target = np.zeros(len(data), dtype=int)
        active = False
        exit_at = -1
        for i, ready in enumerate(confirmed):
            if active:
                if i == exit_at:
                    active = False
                else:
                    target[i] = 1
            elif ready:
                active = True
                exit_at = i + params.hold_bars
                target[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = target
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")
