from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletRecoilParams:
    energy_lookback: int = 126
    energy_percentile: float = 0.80
    atr_period: int = 14
    vol_period: int = 20
    stop_atr: float = 1.5
    target_vol: float = 0.012
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[WaveletRecoilParams]):
    strategy_id = "gen_local_1791298341"

    @classmethod
    def params_type(cls) -> type[WaveletRecoilParams]:
        return WaveletRecoilParams

    @classmethod
    def warmup_bars(cls, params: WaveletRecoilParams) -> int:
        return max(params.energy_lookback + 9, params.atr_period + 1, params.vol_period + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletRecoilParams) -> pd.DataFrame:
        ret = data["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        wave1 = ret - ret.shift(1)
        wave2 = ret.rolling(2).mean() - ret.shift(2).rolling(2).mean()
        wave4 = ret.rolling(4).mean() - ret.shift(4).rolling(4).mean()
        energy1 = wave1.pow(2)
        total_energy = energy1 + wave2.pow(2) + wave4.pow(2)
        fine_share = (energy1 / total_energy.replace(0.0, np.nan)).fillna(0.0)
        threshold = fine_share.rolling(params.energy_lookback, min_periods=params.energy_lookback).quantile(params.energy_percentile).fillna(1.0)
        previous_close = data["close"].shift(1)
        true_range = pd.concat([(data["high"] - data["low"]), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_period, min_periods=params.atr_period).mean().fillna(0.0)
        vol = ret.rolling(params.vol_period, min_periods=params.vol_period).std().fillna(0.02).clip(lower=0.002)
        return pd.DataFrame({"ret": ret, "fine_share": fine_share, "threshold": threshold, "atr": atr, "vol": vol}, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletRecoilParams) -> SignalFrame:
        ret = indicators["ret"]
        share = indicators["fine_share"]
        threshold = indicators["threshold"]
        candidate = ((share.shift(1) > threshold.shift(1)) & (share < share.shift(1)) & (ret * ret.shift(1) < 0)).fillna(False)
        desired = np.zeros(len(data), dtype=int)
        position = 0
        stop = 0.0
        held = 0
        for i in range(len(data)):
            if i and desired[i - 1] != position:
                position = int(desired[i - 1])
                held = 0
                if position:
                    stop = float(data["open"].iat[i]) - position * params.stop_atr * float(indicators["atr"].iat[i - 1])
            if position:
                held += 1
                close = float(data["close"].iat[i])
                stop_hit = close <= stop if position == 1 else close >= stop
                desired[i] = 0 if stop_hit or held >= params.max_hold else position
            elif bool(candidate.iat[i]):
                desired[i] = 1 if ret.iat[i] > 0 else -1
        df = pd.DataFrame(index=data.index)
        df["signal"] = desired
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (params.target_vol / indicators["vol"]).clip(lower=0.01, upper=0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
