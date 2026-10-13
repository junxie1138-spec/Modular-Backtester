from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    cycle_bars: int = 20
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791229448"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.cycle_bars + 1, 11)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.cycle_bars < 10 or params.hold_bars < 1:
            raise ValueError("cycle_bars must be >= 10 and hold_bars must be positive")
        out = pd.DataFrame(index=data.index)
        out["close_change"] = data["close"].diff()
        volume = data["volume"].astype(float).clip(lower=0)
        out["down_volume"] = volume.where(out["close_change"] < 0, 0.0)
        out["directional_volume"] = volume.where(out["close_change"].ne(0) & out["close_change"].notna(), 0.0)
        out["volume_baseline"] = volume.rolling(params.cycle_bars, min_periods=params.cycle_bars).mean().shift(1)
        out["relative_volume"] = volume.div(out["volume_baseline"].replace(0, np.nan))
        out["cycle_down_volume"] = out["down_volume"].rolling(params.cycle_bars, min_periods=params.cycle_bars).sum()
        out["cycle_directional_volume"] = out["directional_volume"].rolling(params.cycle_bars, min_periods=params.cycle_bars).sum()
        out["selling_share"] = out["cycle_down_volume"].div(out["cycle_directional_volume"].replace(0, np.nan))
        out["recent_down_volume"] = out["down_volume"].rolling(5, min_periods=5).sum()
        out["recent_directional_volume"] = out["directional_volume"].rolling(5, min_periods=5).sum()
        out["recent_selling_share"] = out["recent_down_volume"].div(out["recent_directional_volume"].replace(0, np.nan))
        out["prior_selling_share"] = out["recent_selling_share"].shift(5)
        out["quiet_volume_ceiling"] = out["relative_volume"].shift(1).rolling(3, min_periods=3).max()
        out["previous_high"] = data["high"].shift(1)
        out["close_location"] = (data["close"] - data["low"]).div((data["high"] - data["low"]).replace(0, np.nan))
        out["volume_pulse"] = (out["relative_volume"] >= 1.3) & (out["close_change"] > 0) & (out["close_location"] >= 0.75)
        out["recovery_regime"] = out["selling_share"] >= 0.55
        out["expansion_regime"] = out["selling_share"] <= 0.45
        out["recovery_entry"] = out["recovery_regime"] & (out["recent_selling_share"] < out["prior_selling_share"]) & (data["close"] > out["previous_high"])
        out["expansion_entry"] = out["expansion_regime"] & (out["quiet_volume_ceiling"] < 1.0)
        out["entry"] = (out["volume_pulse"] & (out["recovery_entry"] | out["expansion_entry"])).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        positions = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar == exit_bar:
                    exit_bar = -1
                else:
                    positions[bar] = 1
            elif entries[bar]:
                positions[bar] = 1
                exit_bar = bar + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    sample = pd.DataFrame(index=index)
    entries = pd.DataFrame({"entry": True}, index=index)
    result = GeneratedStrategy.generate_signals(None, sample, entries, None, StrategyParams()).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] + [1] * 18 + [0, 1]
    assert result["size"].gt(0).all()
