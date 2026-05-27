from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_window: int = 5
    range_window: int = 20
    atr_window: int = 14
    atr_stop_mult: float = 2.0
    spike_window: int = 60
    spike_pct: float = 0.95
    refractory_bars: int = 5
    max_hold_bars: int = 3
    lower_pos_thresh: float = 0.30
    upper_pos_thresh: float = 0.70


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779880303"

    @classmethod
    def params_type(cls) -> type:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.roc_window + 2,
            params.range_window + 1,
            params.atr_window + 1,
            params.spike_window + params.roc_window + 2,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        roc = close.pct_change(params.roc_window)
        roc_accel = roc.diff()

        hh = high.rolling(params.range_window, min_periods=params.range_window).max()
        ll = low.rolling(params.range_window, min_periods=params.range_window).min()
        rng = (hh - ll).replace(0.0, np.nan)
        rel_pos = (close - ll) / rng

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()

        accel_abs = roc_accel.abs()
        spike_thresh = accel_abs.rolling(
            params.spike_window, min_periods=params.spike_window
        ).quantile(params.spike_pct)
        is_spike = (accel_abs > spike_thresh).fillna(False)

        out = pd.DataFrame(index=data.index)
        out["roc"] = roc
        out["roc_accel"] = roc_accel
        out["rel_pos"] = rel_pos
        out["atr"] = atr
        out["is_spike"] = is_spike.astype(np.int8)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        roc_accel = indicators["roc_accel"]
        rel_pos = indicators["rel_pos"]
        atr = indicators["atr"]
        is_spike = indicators["is_spike"].astype(bool)
        close = data["close"].astype(float)

        pos_accel = (roc_accel > 0).fillna(False)
        neg_accel = (roc_accel < 0).fillna(False)
        pos_accel_prev = pos_accel.shift(1).fillna(False)
        neg_accel_prev = neg_accel.shift(1).fillna(False)

        rel_pos_valid = rel_pos.notna()
        long_setup = (
            pos_accel
            & pos_accel_prev
            & rel_pos_valid
            & (rel_pos < params.lower_pos_thresh)
        )
        short_setup = (
            neg_accel
            & neg_accel_prev
            & rel_pos_valid
            & (rel_pos > params.upper_pos_thresh)
        )

        long_arr = long_setup.to_numpy()
        short_arr = short_setup.to_numpy()
        spike_arr = is_spike.to_numpy()
        close_arr = close.to_numpy()
        atr_arr = atr.to_numpy()

        n = len(data)
        raw = np.zeros(n, dtype=np.int64)

        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0
        cooldown = 0

        for i in range(n):
            if cooldown > 0:
                cooldown -= 1

            if position == 0:
                if cooldown == 0 and not spike_arr[i]:
                    a = atr_arr[i]
                    if not np.isnan(a) and a > 0.0:
                        if long_arr[i]:
                            position = 1
                            entry_price = close_arr[i]
                            entry_atr = a
                            bars_held = 0
                            raw[i] = 1
                        elif short_arr[i]:
                            position = -1
                            entry_price = close_arr[i]
                            entry_atr = a
                            bars_held = 0
                            raw[i] = -1
            else:
                bars_held += 1
                exited = False

                if position == 1:
                    if close_arr[i] <= entry_price - params.atr_stop_mult * entry_atr:
                        exited = True
                else:
                    if close_arr[i] >= entry_price + params.atr_stop_mult * entry_atr:
                        exited = True

                if not exited and bars_held >= params.max_hold_bars:
                    exited = True
                if not exited and spike_arr[i]:
                    exited = True

                if exited:
                    raw[i] = 0
                    position = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    bars_held = 0
                    cooldown = params.refractory_bars
                else:
                    raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = pd.Series(raw, index=data.index).shift(1).fillna(0).astype(int)
        df["size"] = 1.0

        return SignalFrame(data=df, signal_column="signal", size_column="size")
