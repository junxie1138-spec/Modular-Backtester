from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FrontierParams:
    energy_window: int = 10
    anchor_window: int = 30
    energy_ratio: float = 1.8
    displacement_threshold: float = 1.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[FrontierParams]):
    strategy_id = "gen_local_1791209952"

    @classmethod
    def params_type(cls) -> type[FrontierParams]:
        return FrontierParams

    def warmup_bars(self, params: FrontierParams) -> int:
        return max(params.energy_window + 1, params.anchor_window)

    def indicators(self, data: pd.DataFrame, params: FrontierParams) -> pd.DataFrame:
        if params.energy_window < 2 or params.anchor_window < 2:
            raise ValueError("Lookback windows must be at least two bars")
        if not np.isfinite(params.energy_ratio) or params.energy_ratio <= 1:
            raise ValueError("energy_ratio must exceed one")
        if not np.isfinite(params.displacement_threshold) or params.displacement_threshold <= 0:
            raise ValueError("displacement_threshold must be positive")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")

        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["lower_extension"] = (-data["low"].diff()).clip(lower=0)
        out["upper_extension"] = data["high"].diff().clip(lower=0)
        out["lower_energy"] = out["lower_extension"].pow(2).rolling(params.energy_window).sum()
        out["upper_energy"] = out["upper_extension"].pow(2).rolling(params.energy_window).sum()
        out["volume_sum"] = data["volume"].clip(lower=0).rolling(params.anchor_window).sum()
        out["weighted_close_sum"] = (data["close"] * data["volume"].clip(lower=0)).rolling(params.anchor_window).sum()
        out["anchor"] = out["weighted_close_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["mean_range"] = out["range"].rolling(params.anchor_window).mean()
        out["displacement"] = (data["close"] - out["anchor"]) / out["mean_range"].where(out["mean_range"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: FrontierParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        lower = indicators["lower_energy"]
        upper = indicators["upper_energy"]
        displacement = indicators["displacement"]
        ready = pd.Series(np.isfinite(indicators[["lower_energy", "upper_energy", "displacement"]].to_numpy()).all(axis=1), index=data.index)
        long_entry = ready & (lower > params.energy_ratio * upper) & (displacement <= -params.displacement_threshold)
        short_entry = ready & (upper > params.energy_ratio * lower) & (displacement >= params.displacement_threshold)
        candidates = np.select([long_entry.to_numpy(), short_entry.to_numpy()], [1, -1], default=0)

        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signals[i] = position
                continue
            if candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = FrontierParams()
    index = pd.date_range("2020-01-01", periods=42, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    evidence = pd.DataFrame({"lower_energy": 4.0, "upper_energy": 1.0, "displacement": -2.0}, index=index)
    evidence.iloc[4:20] = [1.0, 4.0, 2.0]
    evidence.iloc[0] = np.nan
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, evidence, None, params).data
    assert result["signal"].iloc[:2].eq(0).all()
    assert result["signal"].iloc[2:20].eq(1).all()
    assert result["signal"].iloc[20] == 0
    assert result["signal"].iloc[21:39].eq(-1).all()
    assert result["signal"].iloc[39] == 0
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) == 30
