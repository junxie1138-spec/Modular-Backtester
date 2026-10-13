from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entry_z: float = 0.7
    trail_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791254220"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 41

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(40, min_periods=40).mean()
        out["dispersion"] = close.rolling(40, min_periods=40).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["dispersion"].replace(0.0, np.nan)
        out["returns"] = close.pct_change(fill_method=None)
        out["squared_returns"] = out["returns"].pow(2)
        out["downside_squared"] = out["returns"].clip(upper=0.0).pow(2)
        out["total_power"] = out["squared_returns"].rolling(20, min_periods=20).mean()
        out["downside_power"] = out["downside_squared"].rolling(20, min_periods=20).mean()
        out["downside_share"] = out["downside_power"] / out["total_power"].replace(0.0, np.nan)
        out["volatility"] = out["returns"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252.0)
        out["previous_close"] = close.shift(1)
        out["intraday_range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["intraday_range", "high_gap", "low_gap"]].max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["previous_z"] = out["z"].shift(1)
        out["ready"] = (
            np.isfinite(out[["z", "previous_z", "downside_share", "volatility", "atr"]]).all(axis=1)
            & (out["volatility"] > 0.0)
            & (out["atr"] > 0.0)
            & np.isfinite(close)
            & (close > 0.0)
        )
        out["entry"] = (
            out["ready"]
            & (out["z"] >= params.entry_z)
            & (out["previous_z"] < params.entry_z)
            & (out["downside_share"] <= 0.35)
        ).fillna(False)
        out["weight"] = (0.10 / out["volatility"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95).fillna(0.95)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if active:
                # The first active iteration is the entry-fill bar.
                age += 1
                if not ready[i]:
                    active = False
                else:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                    if closes[i] <= stop or age >= 18:
                        active = False
                signals[i] = int(active)
                continue
            if ready[i] and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["weight"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    sample = pd.DataFrame({"close": [100.0, 102.0, 100.0, 98.0, 98.0]}, index=index)
    features = pd.DataFrame({
        "atr": [1.0, 1.0, 10.0, 10.0, 10.0],
        "entry": [True, False, False, False, False],
        "ready": [True] * 5,
        "weight": [0.5] * 5,
    }, index=index)
    result = GeneratedStrategy.generate_signals(sample, features, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert (result["size"] > 0.0).all()
