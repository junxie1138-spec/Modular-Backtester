from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    roc_window: int = 5
    accel_lookback: int = 3
    autocorr_window: int = 20
    atr_window: int = 14
    atr_mult: float = 2.0
    max_hold_bars: int = 2
    accel_threshold: float = 0.0
    autocorr_threshold: float = 0.05


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a1_1779880440"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.autocorr_window + params.roc_window + 5,
            params.atr_window + 5,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        roc = close.pct_change(params.roc_window)
        roc_accel = roc.diff(params.accel_lookback)

        returns = close.pct_change()
        autocorr = returns.rolling(params.autocorr_window).corr(returns.shift(1))

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_window).mean()

        return pd.DataFrame({
            "roc_accel": roc_accel,
            "autocorr": autocorr,
            "atr": atr,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        close = data["close"].to_numpy()
        roc_accel = indicators["roc_accel"].to_numpy()
        autocorr = indicators["autocorr"].to_numpy()
        atr = indicators["atr"].to_numpy()

        raw_signal = np.zeros(n, dtype=np.int64)

        accel_ok = np.isfinite(roc_accel) & (roc_accel > params.accel_threshold)
        autocorr_ok = np.isfinite(autocorr) & (autocorr > params.autocorr_threshold)
        atr_ok = np.isfinite(atr) & (atr > 0.0)
        entry_cond = accel_ok & autocorr_ok & atr_ok

        in_pos = False
        entry_price = 0.0
        entry_atr = 0.0
        bars_held = 0

        for i in range(n):
            if in_pos:
                bars_held += 1
                stop_price = entry_price - params.atr_mult * entry_atr
                if not np.isfinite(close[i]):
                    in_pos = False
                    bars_held = 0
                    raw_signal[i] = 0
                elif close[i] < stop_price or bars_held >= params.max_hold_bars:
                    in_pos = False
                    bars_held = 0
                    raw_signal[i] = 0
                else:
                    raw_signal[i] = 1
            else:
                if entry_cond[i]:
                    in_pos = True
                    entry_price = float(close[i])
                    entry_atr = float(atr[i])
                    bars_held = 0
                    raw_signal[i] = 1
                else:
                    raw_signal[i] = 0

        df = pd.DataFrame({
            "signal": raw_signal,
            "size": np.ones(n, dtype=np.float64),
        }, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
