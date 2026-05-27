from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rv_window: int = 20
    rv_z_window: int = 252
    vol_z_low: float = 0.5
    vol_z_high: float = 2.0
    profit_target: float = 0.04
    time_stop_bars: int = 8
    min_seasonal_obs: int = 24


class GeneratedStrategy(BaseStrategy):
    strategy_id = "gen_a1_1779647038"

    @classmethod
    def params_type(cls):
        return Params

    @classmethod
    def warmup_bars(cls, params: Params) -> int:
        return int(params.rv_z_window + params.rv_window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        ret = close.pct_change()

        rv_window = int(params.rv_window)
        z_window = int(params.rv_z_window)

        rv = ret.rolling(rv_window, min_periods=rv_window).std()
        rv_mean = rv.rolling(z_window, min_periods=z_window).mean()
        rv_std = rv.rolling(z_window, min_periods=z_window).std()
        rv_z = (rv - rv_mean) / rv_std.replace(0.0, np.nan)

        dom_values = np.asarray(data.index.day, dtype=np.int64)
        dom = pd.Series(dom_values, index=data.index, dtype="int32")

        seasonal = pd.Series(np.nan, index=data.index, dtype="float64")
        seasonal_count = pd.Series(np.nan, index=data.index, dtype="float64")
        for d in range(1, 32):
            mask = dom_values == d
            if not mask.any():
                continue
            sub = ret.iloc[mask]
            seasonal.iloc[mask] = sub.expanding(min_periods=1).mean().values
            seasonal_count.iloc[mask] = sub.expanding(min_periods=1).count().values

        ind = pd.DataFrame(index=data.index)
        ind["ret"] = ret
        ind["rv"] = rv
        ind["rv_z"] = rv_z
        ind["dom"] = dom
        ind["seasonal"] = seasonal
        ind["seasonal_count"] = seasonal_count
        ind["close"] = close
        return ind

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        rv_z = indicators["rv_z"]
        seasonal = indicators["seasonal"]
        seasonal_count = indicators["seasonal_count"]
        close = indicators["close"]

        vol_low = float(params.vol_z_low)
        vol_high = float(params.vol_z_high)
        min_obs = float(params.min_seasonal_obs)

        elastic = (rv_z >= vol_low) & (rv_z <= vol_high)
        seasonal_ok = (seasonal > 0.0) & (seasonal_count >= min_obs)
        setup = (elastic & seasonal_ok).fillna(False).astype(bool)
        confirmed = (setup & setup.shift(1).fillna(False).astype(bool)).fillna(False).astype(bool)

        n = len(data)
        signal = np.zeros(n, dtype=np.int64)
        close_v = close.values
        confirmed_v = confirmed.values

        in_pos = False
        entry_price = 0.0
        bars_held = 0
        profit_target = float(params.profit_target)
        max_bars = int(params.time_stop_bars)

        for i in range(n):
            px_finite = np.isfinite(close_v[i])
            if not in_pos:
                if confirmed_v[i] and px_finite:
                    in_pos = True
                    entry_price = float(close_v[i])
                    bars_held = 0
                    signal[i] = 1
            else:
                bars_held += 1
                px = float(close_v[i]) if px_finite else entry_price
                gain = (px / entry_price) - 1.0 if entry_price > 0.0 else 0.0
                if gain >= profit_target or bars_held >= max_bars:
                    in_pos = False
                    entry_price = 0.0
                    bars_held = 0
                    signal[i] = 0
                else:
                    signal[i] = 1

        out = pd.DataFrame(index=data.index)
        out["signal"] = pd.Series(signal, index=data.index).shift(1).fillna(0).astype(int)
        out["size"] = 1.0
        return SignalFrame(data=out, signal_column="signal", size_column="size")
