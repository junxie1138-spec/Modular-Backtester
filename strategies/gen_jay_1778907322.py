from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolSurgeHysteresisParams:
    vol_lookback: int = 20
    vol_arm_z: float = 1.0
    vol_fire_z: float = 1.8
    eff_lookback: int = 5
    eff_arm_thresh: float = 0.15
    eff_fire_thresh: float = 0.28
    atr_lookback: int = 14
    atr_stop_k: float = 2.0
    target_daily_vol: float = 0.015
    cooldown_bars: int = 2


class GeneratedStrategy(BaseStrategy[VolSurgeHysteresisParams]):
    strategy_id = "gen_jay_1778907322"

    @classmethod
    def params_type(cls):
        return VolSurgeHysteresisParams

    @staticmethod
    def warmup_bars(params: VolSurgeHysteresisParams) -> int:
        return max(params.vol_lookback, params.atr_lookback) + params.eff_lookback + 5

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolSurgeHysteresisParams) -> pd.DataFrame:
        ind = pd.DataFrame(index=data.index)
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        open_ = data["open"]

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.ewm(span=params.atr_lookback, min_periods=params.atr_lookback).mean()

        vol_mean = volume.rolling(params.vol_lookback, min_periods=params.vol_lookback).mean()
        vol_std = volume.rolling(params.vol_lookback, min_periods=params.vol_lookback).std()
        raw_z = (volume - vol_mean) / (vol_std + 1e-10)
        bar_dir = np.sign(close - open_)
        ind["signed_vol_z"] = raw_z * bar_dir

        bar_range = (high - low).clip(lower=1e-8)
        dir_eff = (close - open_) / bar_range
        ind["dir_eff"] = dir_eff.rolling(params.eff_lookback, min_periods=params.eff_lookback).mean()

        ret = close.pct_change()
        ret_std = ret.rolling(params.vol_lookback, min_periods=params.vol_lookback).std()
        ind["vol_scalar"] = (params.target_daily_vol / (ret_std + 1e-10)).clip(0.1, 2.0)

        return ind

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolSurgeHysteresisParams,
    ) -> SignalFrame:
        n = len(data)
        close = data["close"].values
        signed_vol_z = indicators["signed_vol_z"].values
        dir_eff = indicators["dir_eff"].values
        atr = indicators["atr"].values
        vol_scalar = indicators["vol_scalar"].values

        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.95, dtype=float)

        position = 0
        armed = 0
        entry_price = np.nan
        cooldown = 0

        for i in range(n):
            sv = signed_vol_z[i]
            de = dir_eff[i]
            c = close[i]
            a = atr[i]
            vs = vol_scalar[i]

            if np.isnan(sv) or np.isnan(de) or np.isnan(a) or np.isnan(vs):
                signal[i] = 0
                size[i] = 0.95
                continue

            # Fixed volatility stop: compare bar close to entry_price +/- k*ATR
            if position == 1 and not np.isnan(entry_price):
                if c < entry_price - params.atr_stop_k * a:
                    position = 0
                    armed = 0
                    entry_price = np.nan
                    cooldown = params.cooldown_bars
            elif position == -1 and not np.isnan(entry_price):
                if c > entry_price + params.atr_stop_k * a:
                    position = 0
                    armed = 0
                    entry_price = np.nan
                    cooldown = params.cooldown_bars

            if cooldown > 0:
                cooldown -= 1

            if position == 0 and cooldown == 0:
                # Fire: requires armed state from prior bar AND fire-threshold met now
                if armed == 1 and sv >= params.vol_fire_z and de >= params.eff_fire_thresh:
                    position = 1
                    entry_price = c
                    armed = 0
                elif armed == -1 and sv <= -params.vol_fire_z and de <= -params.eff_fire_thresh:
                    position = -1
                    entry_price = c
                    armed = 0
                else:
                    # Arm for next bar if arm-threshold met in both primitives
                    if sv >= params.vol_arm_z and de >= params.eff_arm_thresh:
                        armed = 1
                    elif sv <= -params.vol_arm_z and de <= -params.eff_arm_thresh:
                        armed = -1
                    else:
                        armed = 0

            signal[i] = position
            size[i] = float(np.clip(vs, 0.1, 2.0))

        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
