from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRenewalParams:
    renewal_window: int = 15
    volatility_window: int = 20
    atr_window: int = 14
    entry_balance: float = 0.25
    min_renewal: float = 0.15
    target_volatility: float = 0.12
    trailing_atr: float = 3.0
    max_hold: int = 20


class GeneratedStrategy(BaseStrategy[RangeRenewalParams]):
    strategy_id = "gen_local-w2_1790989579"

    @classmethod
    def params_type(cls):
        return RangeRenewalParams

    def warmup_bars(self, params):
        return int(max(params.renewal_window + 1,
                       params.volatility_window, params.atr_window + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        out["range"] = (high - low).clip(lower=0.0)
        out["up_renewal"] = (high - high.shift(1)).clip(lower=0.0).clip(upper=out["range"])
        out["down_renewal"] = (low.shift(1) - low).clip(lower=0.0).clip(upper=out["range"])
        out["up_mass"] = out["up_renewal"].rolling(params.renewal_window).sum()
        out["down_mass"] = out["down_renewal"].rolling(params.renewal_window).sum()
        out["range_mass"] = out["range"].rolling(params.renewal_window).sum()
        out["renewed_mass"] = out["up_mass"] + out["down_mass"]
        out["balance"] = ((out["up_mass"] - out["down_mass"]) /
                          out["renewed_mass"].replace(0.0, np.nan))
        out["renewal_fraction"] = (out["renewed_mass"] /
                                   out["range_mass"].replace(0.0, np.nan))
        out["log_range_squared"] = np.log(high.where(high > 0.0) /
                                          low.where(low > 0.0)).pow(2)
        out["annual_volatility"] = np.sqrt(
            out["log_range_squared"].rolling(params.volatility_window).mean()
            * 252.0 / (4.0 * np.log(2.0)))
        out["true_range"] = pd.concat(
            [out["range"], (high - close.shift(1)).abs(),
             (low - close.shift(1)).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["strength"] = out["balance"].clip(lower=0.0, upper=1.0)
        out["target_size"] = (
            params.target_volatility * out["strength"] /
            out["annual_volatility"].replace(0.0, np.nan)
        ).clip(lower=0.05, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        n = len(data)
        signal = np.zeros(n, dtype=int)
        size = np.full(n, 0.05, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        target = indicators["target_size"].to_numpy(dtype=float)
        ready = (
            indicators[["balance", "renewal_fraction", "annual_volatility",
                        "atr", "target_size"]].notna().all(axis=1)
            & indicators["atr"].gt(0.0)
            & indicators["annual_volatility"].gt(0.0)
        ).to_numpy()
        enter = (
            indicators["balance"].ge(params.entry_balance)
            & indicators["renewal_fraction"].ge(params.min_renewal)
        ).fillna(False).to_numpy()
        active = False
        age = 0
        high_water = np.nan
        stop = np.nan
        for i in range(n):
            if not ready[i] or not np.isfinite(close[i]) or close[i] <= 0.0:
                active = False
                age = 0
                high_water = np.nan
                stop = np.nan
                continue
            if active:
                age += 1
                # The first active iteration is the actual next-bar entry.
                high_water = close[i] if not np.isfinite(high_water) else max(high_water, close[i])
                candidate = high_water - params.trailing_atr * atr[i]
                stop = candidate if not np.isfinite(stop) else max(stop, candidate)
                if close[i] <= stop or age >= params.max_hold:
                    active = False
                    age = 0
                    high_water = np.nan
                    stop = np.nan
                    continue
            elif enter[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = np.nan
            if active:
                signal[i] = 1
                size[i] = target[i]
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
