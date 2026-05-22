from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    flow_window: int = 5
    vol_window: int = 9
    volume_avg_window: int = 10
    tide_threshold: float = 0.35
    volume_mult: float = 1.2
    target_vol: float = 0.02
    min_size: float = 0.3
    max_size: float = 2.0
    profit_target: float = 0.03
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779147613"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(
            max(
                int(params.flow_window) + 1,
                int(params.vol_window) + 1,
                int(params.volume_avg_window),
            )
        )

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        # Tick-rule signed volume: net money flow over a short window,
        # normalized to the fraction of total volume that was net buying.
        signed = np.sign(close.diff()).fillna(0.0)
        flow = signed * volume
        w = max(int(params.flow_window), 1)
        tide = flow.rolling(w, min_periods=w).sum()
        vol_sum = volume.rolling(w, min_periods=w).sum()
        tide_norm = (tide / vol_sum.replace(0.0, np.nan)).fillna(0.0)

        vavg_w = max(int(params.volume_avg_window), 1)
        vol_avg = volume.rolling(vavg_w, min_periods=1).mean()

        rv_w = max(int(params.vol_window), 2)
        realized_vol = close.pct_change().rolling(rv_w, min_periods=2).std()

        out = pd.DataFrame(index=data.index)
        out["tide_norm"] = tide_norm
        out["vol_avg"] = vol_avg
        out["realized_vol"] = realized_vol
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        tide = indicators["tide_norm"].fillna(0.0)
        vol_avg = indicators["vol_avg"]
        rvol = indicators["realized_vol"]

        up_bar = close > close.shift(1)
        vol_surge = volume > (vol_avg * float(params.volume_mult))
        rising_tide = tide > float(params.tide_threshold)
        raw_entry = (up_bar & vol_surge & rising_tide).fillna(False).to_numpy()

        c = close.to_numpy(dtype=float)
        n = len(c)
        sig = np.zeros(n, dtype=int)
        in_pos = False
        entry_price = 0.0
        held = 0
        pt = float(params.profit_target)
        mh = max(int(params.max_hold), 1)

        for i in range(n):
            if in_pos:
                held += 1
                ret = (c[i] / entry_price - 1.0) if entry_price > 0.0 else 0.0
                if ret >= pt or held >= mh:
                    sig[i] = 0
                    in_pos = False
                else:
                    sig[i] = 1
            else:
                if raw_entry[i]:
                    in_pos = True
                    entry_price = c[i]
                    held = 0
                    sig[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        tv = float(params.target_vol)
        size = tv / rvol.replace(0.0, np.nan)
        size = size.clip(lower=float(params.min_size), upper=float(params.max_size))
        size = size.fillna(float(params.min_size))
        df["size"] = size.astype(float)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
