from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    trend_len: int = 100
    corr_len: int = 20
    coh_enter: float = 0.25
    roc_len: int = 10
    atr_len: int = 14
    atr_mult: float = 3.0
    size_min: float = 0.40
    size_max: float = 1.00


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_a2_1779148787"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.trend_len, params.corr_len + 1,
                       params.roc_len, params.atr_len)) + 5

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        ind = pd.DataFrame(index=data.index)

        # Trend regime gate.
        ind["sma"] = close.rolling(params.trend_len, min_periods=params.trend_len).mean()

        # Trend-strength magnitude over the holding horizon.
        ind["roc"] = close.pct_change(params.roc_len)

        # Volume-price coherence: rolling correlation between daily returns and
        # the change in volume. A positive coherence means volume expands on
        # up days and contracts on down days -> the move is volume-confirmed.
        ret = close.pct_change()
        vol_chg = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        ind["coh"] = ret.rolling(params.corr_len, min_periods=params.corr_len).corr(vol_chg)

        # ATR for the trailing stop.
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(params.atr_len, min_periods=params.atr_len).mean()

        return ind

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        sma = indicators["sma"].to_numpy(dtype=float)
        roc = indicators["roc"].to_numpy(dtype=float)
        coh = indicators["coh"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)

        n = len(close)
        coh_prev = np.empty(n, dtype=float)
        coh_prev[0] = np.nan
        coh_prev[1:] = coh[:-1]

        # Entry: trend up, positive horizon momentum, and volume-price coherence
        # crossing up through the threshold (the move just became volume-confirmed).
        entry = (
            (close > sma)
            & (roc > 0.0)
            & (coh > params.coh_enter)
            & (coh_prev <= params.coh_enter)
        )
        entry = np.where(np.isnan(entry), False, entry)

        # Signal-scaled sizing: deeper coherence -> larger position.
        denom = max(1.0 - params.coh_enter, 1e-6)
        depth = np.clip((coh - params.coh_enter) / denom, 0.0, 1.0)
        depth = np.where(np.isnan(depth), 0.0, depth)
        size_arr = params.size_min + depth * (params.size_max - params.size_min)

        sig = np.zeros(n, dtype=int)
        size_out = np.full(n, params.size_min, dtype=float)

        position = 0
        hwm = np.nan
        entry_size = params.size_min

        for i in range(n):
            if position == 0:
                if entry[i] and not np.isnan(atr[i]):
                    position = 1
                    hwm = close[i]
                    entry_size = float(size_arr[i])
                    sig[i] = 1
                    size_out[i] = entry_size
            else:
                if close[i] > hwm:
                    hwm = close[i]
                stop = hwm - params.atr_mult * atr[i]
                if np.isnan(atr[i]) or close[i] < stop:
                    position = 0
                    hwm = np.nan
                    sig[i] = 0
                else:
                    sig[i] = 1
                    size_out[i] = entry_size

        df = pd.DataFrame(index=data.index)
        df["signal"] = sig
        df["size"] = size_out
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.size_min)
        df["size"] = df["size"].clip(lower=params.size_min, upper=params.size_max)

        return SignalFrame(data=df, signal_column="signal", size_column="size")
