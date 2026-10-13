from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockClearanceParams:
    entry_z: float = 1.1
    target_vol: float = 0.10


class GeneratedStrategy(BaseStrategy[ShockClearanceParams]):
    strategy_id = "gen_local_1791333564"

    @classmethod
    def params_type(cls) -> type[ShockClearanceParams]:
        return ShockClearanceParams

    def warmup_bars(self, params: ShockClearanceParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: ShockClearanceParams) -> pd.DataFrame:
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["ma"] = close.rolling(20, min_periods=20).mean()
        out["price_std"] = close.rolling(20, min_periods=20).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["price_std"].replace(0.0, np.nan)
        out["return"] = close.pct_change(fill_method=None)
        out["realized_vol"] = out["return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        out["downside_energy"] = out["return"].clip(upper=0.0).pow(2)
        out["recent_shock"] = out["downside_energy"].rolling(5, min_periods=5).max()
        out["total_downside_energy"] = out["downside_energy"].rolling(20, min_periods=20).sum()
        out["shock_share"] = out["recent_shock"] / out["total_downside_energy"].replace(0.0, np.nan)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            (out["z"].shift(1) < -params.entry_z)
            & (out["z"] >= -params.entry_z)
            & (out["z"] < 0.0)
            & (out["shock_share"] >= 0.50)
            & (out["return"] > 0.0)
        )
        out["exposure"] = (params.target_vol / out["realized_vol"].replace(0.0, np.nan)).clip(0.05, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockClearanceParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        exposure = indicators["exposure"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = 0.0
        stop = 0.0
        held_size = 0.05
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0
            if active:
                age += 1
                if not valid:
                    active = False
                    continue
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - 2.0 * atr[i])
                if close[i] <= stop or age >= 4:
                    active = False
                    continue
            elif entry[i] and valid and np.isfinite(exposure[i]):
                active = True
                age = 0
                high_water = close[i]
                stop = high_water - 2.0 * atr[i]
                held_size = float(exposure[i])
            if active:
                signals[i] = 1
                sizes[i] = held_size
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        assert df["signal"].isin([0, 1]).all() and (df["size"] > 0.0).all()
        return SignalFrame(data=df, signal_column="signal", size_column="size")
