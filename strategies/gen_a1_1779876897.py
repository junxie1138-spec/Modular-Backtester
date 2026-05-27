from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlasticGapParams:
    component_window: int = 10
    gap_z_threshold: float = 0.5
    intraday_z_threshold: float = 0.5
    z_window: int = 60
    atr_window: int = 20
    trail_atr_mult: float = 3.0
    regime_ma: int = 200


class GeneratedStrategy(BaseStrategy[PlasticGapParams]):
    strategy_id = "gen_a1_1779876897"

    @classmethod
    def params_type(cls):
        return PlasticGapParams

    def warmup_bars(self, params: PlasticGapParams) -> int:
        return int(max(params.regime_ma, params.z_window + params.component_window)) + 5

    def indicators(self, data: pd.DataFrame, params: PlasticGapParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        prev_close = close.shift(1)

        safe_prev = prev_close.where(prev_close > 0)
        safe_open = open_.where(open_ > 0)

        gap_ret = np.log(safe_open / safe_prev)
        intraday_ret = np.log(close.where(close > 0) / safe_open)

        cw = int(max(2, params.component_window))
        zw = int(max(5, params.z_window))

        gap_sum = gap_ret.rolling(cw, min_periods=cw).sum()
        intraday_sum = intraday_ret.rolling(cw, min_periods=cw).sum()

        gap_mean = gap_sum.rolling(zw, min_periods=zw).mean()
        gap_std = gap_sum.rolling(zw, min_periods=zw).std(ddof=0).replace(0.0, np.nan)
        gap_z = (gap_sum - gap_mean) / gap_std

        intraday_mean = intraday_sum.rolling(zw, min_periods=zw).mean()
        intraday_std = intraday_sum.rolling(zw, min_periods=zw).std(ddof=0).replace(0.0, np.nan)
        intraday_z = (intraday_sum - intraday_mean) / intraday_std

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(int(max(2, params.atr_window)), min_periods=int(max(2, params.atr_window))).mean()

        regime_ma = close.rolling(int(max(2, params.regime_ma)), min_periods=int(max(2, params.regime_ma))).mean()

        return pd.DataFrame({
            "gap_z": gap_z,
            "intraday_z": intraday_z,
            "atr": atr,
            "regime_ma": regime_ma,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: PlasticGapParams) -> SignalFrame:
        close = data["close"].astype(float)
        gap_z = indicators["gap_z"]
        intraday_z = indicators["intraday_z"]
        atr = indicators["atr"]
        regime_ma = indicators["regime_ma"]

        entry_cond = (
            (gap_z > float(params.gap_z_threshold))
            & (intraday_z > float(params.intraday_z_threshold))
            & (close > regime_ma)
            & atr.notna()
            & (atr > 0)
        )
        entry_cond = entry_cond.fillna(False)

        close_arr = close.to_numpy(dtype=float)
        atr_arr = atr.to_numpy(dtype=float)
        entry_arr = entry_cond.to_numpy(dtype=bool)

        n = len(close_arr)
        signal_raw = np.zeros(n, dtype=np.int64)
        in_position = False
        high_water = np.nan
        k = float(params.trail_atr_mult)

        for i in range(n):
            if in_position:
                if not np.isnan(close_arr[i]) and (np.isnan(high_water) or close_arr[i] > high_water):
                    high_water = close_arr[i]
                stop_level = high_water - k * atr_arr[i] if not np.isnan(atr_arr[i]) else np.nan
                if not np.isnan(stop_level) and close_arr[i] <= stop_level:
                    in_position = False
                    high_water = np.nan
                    signal_raw[i] = 0
                else:
                    signal_raw[i] = 1
            else:
                if entry_arr[i]:
                    in_position = True
                    high_water = close_arr[i]
                    signal_raw[i] = 1
                else:
                    signal_raw[i] = 0

        signal_series = pd.Series(signal_raw, index=data.index)
        signal = signal_series.shift(1).fillna(0).astype(int)
        size = pd.Series(1.0, index=data.index)

        out = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        return SignalFrame(data=out, signal_column="signal", size_column="size")
