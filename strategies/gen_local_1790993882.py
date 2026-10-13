from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveParams:
    fractional_order: float = 0.45
    stop_atr: float = 1.25


class GeneratedStrategy(BaseStrategy[WaveParams]):
    strategy_id = "gen_local_1790993882"

    @classmethod
    def params_type(cls) -> type[WaveParams]:
        return WaveParams

    def warmup_bars(self, params: WaveParams) -> int:
        return 72

    def indicators(self, data: pd.DataFrame, params: WaveParams) -> pd.DataFrame:
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        # Fixed-width causal fractional difference.
        weights = np.ones(32, dtype=float)
        for j in range(1, 32):
            weights[j] = -weights[j - 1] * (params.fractional_order - j + 1) / j
        fd = close * weights[0]
        for j in range(1, 32):
            fd = fd + close.shift(j) * weights[j]
        out["fractional_close"] = fd
        out["displacement"] = fd - fd.rolling(20, min_periods=20).mean()
        out["velocity"] = out["displacement"].diff()
        out["displacement_scale"] = out["displacement"].rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        out["velocity_scale"] = out["velocity"].rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        out["potential"] = (out["displacement"] / out["displacement_scale"]) ** 2
        out["kinetic"] = (out["velocity"] / out["velocity_scale"]) ** 2
        out["energy_gap"] = out["kinetic"] - out["potential"]
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["return"] = close.pct_change(fill_method=None)
        out["annual_vol"] = out["return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        out["target_size"] = (0.10 / out["annual_vol"].replace(0.0, np.nan)).clip(0.10, 0.95)
        out["launch"] = (
            (out["displacement"] < 0.0)
            & (out["velocity"] > 0.0)
            & (out["energy_gap"] > 0.0)
            & (out["energy_gap"].shift(1) <= 0.0)
            & (out["atr"] > 0.0)
            & np.isfinite(out["target_size"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveParams,
    ) -> SignalFrame:
        n = len(data)
        desired = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.10, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        targets = indicators["target_size"].to_numpy(dtype=float)
        launches = indicators["launch"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        fixed_stop = np.nan
        held_size = 0.10
        for i in range(n):
            # Yesterday's entry decision fills at today's open.
            if not active and i > 0 and desired[i - 1] == 1:
                if not (np.isfinite(opens[i]) and np.isfinite(atr[i - 1]) and opens[i] > 0.0):
                    continue
                active = True
                entry_bar = i
                fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
                held_size = sizes[i - 1]
            if active:
                # The entry ATR stop stays fixed. Exit decisions fill next bar.
                if not np.isfinite(closes[i]) or closes[i] < fixed_stop or i - entry_bar + 1 >= 2:
                    active = False
                    continue
                desired[i] = 1
                sizes[i] = held_size
            elif launches[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                desired[i] = 1
                sizes[i] = float(targets[i])
        df = pd.DataFrame({"signal": desired, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.10).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
