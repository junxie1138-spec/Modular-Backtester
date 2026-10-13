from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolatilityParams:
    baseline_bars: int = 60
    shock_bars: int = 6
    shock_multiple: float = 1.5
    imbalance_min: float = 0.6
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[VolatilityParams]):
    strategy_id = "gen_local_1790970621"

    @classmethod
    def params_type(cls):
        return VolatilityParams

    @staticmethod
    def warmup_bars(params: VolatilityParams) -> int:
        return int(params.baseline_bars + params.shock_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: VolatilityParams) -> pd.DataFrame:
        if params.baseline_bars < 2 or params.shock_bars < 2:
            raise ValueError("Volatility windows must be at least two bars")
        if not np.isfinite(params.shock_multiple) or params.shock_multiple <= 0:
            raise ValueError("shock_multiple must be finite and positive")
        if not np.isfinite(params.imbalance_min) or not 0 < params.imbalance_min < 1:
            raise ValueError("imbalance_min must lie between zero and one")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["up_energy"] = out["return"].clip(lower=0).pow(2).rolling(params.shock_bars).mean()
        out["down_energy"] = out["return"].clip(upper=0).pow(2).rolling(params.shock_bars).mean()
        out["total_energy"] = out["up_energy"] + out["down_energy"]
        out["recent_rms"] = np.sqrt(out["total_energy"])
        out["baseline_vol"] = out["return"].rolling(params.baseline_bars).std(ddof=0).shift(params.shock_bars)
        out["burst"] = out["recent_rms"] / out["baseline_vol"].where(out["baseline_vol"] > 0)
        out["imbalance"] = (out["up_energy"] - out["down_energy"]) / out["total_energy"].where(out["total_energy"] > 0)
        out["displacement"] = close / close.shift(params.shock_bars) - 1.0
        out = out.replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            out[["burst", "imbalance", "displacement"]].notna().all(axis=1)
            & out["burst"].ge(params.shock_multiple)
            & out["imbalance"].abs().ge(params.imbalance_min)
            & (out["imbalance"] * out["displacement"]).gt(0)
        )
        out["entry_signal"] = np.where(out["eligible"], -np.sign(out["imbalance"].fillna(0)), 0).astype(int)
        out["conviction"] = (
            0.35 * (out["burst"] / params.shock_multiple)
            * (out["imbalance"].abs() / params.imbalance_min)
        ).clip(lower=0.35, upper=0.95).fillna(0.35)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: VolatilityParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        entries = indicators["entry_signal"].fillna(0).to_numpy(dtype=int)
        strengths = indicators["conviction"].replace([np.inf, -np.inf], np.nan).fillna(0.35).clip(0.35, 0.95).to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.35, dtype=float)
        position = 0
        exit_bar = -1
        entry_size = 0.35
        for i in range(len(data)):
            if position != 0 and i >= exit_bar:
                position = 0
                continue
            if position == 0 and entries[i] != 0:
                position = int(entries[i])
                entry_size = float(strengths[i])
                exit_bar = i + params.hold_bars
            if position != 0:
                signals[i] = position
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.35).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = VolatilityParams()
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({"entry_signal": -1, "conviction": 0.8}, index=index)
    features.iloc[0] = [1, 0.5]
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["size"].iloc[1:19].eq(0.5).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20] == -1
    assert result["size"].iloc[20] == 0.8
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
