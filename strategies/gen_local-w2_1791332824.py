from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 5
    coarse_share: float = 0.65
    upward_share: float = 0.70


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791332824"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return 16 + params.energy_window

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 1:
            raise ValueError("energy_window must be positive")
        if not (0.0 < params.coarse_share < 1.0 and 0.0 < params.upward_share <= 1.0):
            raise ValueError("Energy thresholds must be valid fractions")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        for half in (2, 4, 8):
            mean_key = f"mean_{half}"
            haar_key = f"haar_{half}"
            out[mean_key] = out["log_close"].rolling(half, min_periods=half).mean()
            # Causal, unit-norm Haar filter: recent block minus preceding block.
            out[haar_key] = np.sqrt(half / 2.0) * (out[mean_key] - out[mean_key].shift(half))
            out[f"energy_{half}"] = out[haar_key].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            out[f"up_energy_{half}"] = out[haar_key].clip(lower=0.0).pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["coarse_energy"] = out["energy_4"] + out["energy_8"]
        out["total_energy"] = out["energy_2"] + out["coarse_energy"]
        out["coarse_share"] = out["coarse_energy"] / out["total_energy"].replace(0.0, np.nan)
        out["upward_share"] = (
            out["up_energy_4"] + out["up_energy_8"]
        ) / out["coarse_energy"].replace(0.0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        entry = (
            indicators["coarse_share"].ge(params.coarse_share)
            & indicators["upward_share"].ge(params.upward_share)
            & indicators["haar_8"].gt(0.0)
        ).fillna(False)
        df = pd.DataFrame(index=data.index)
        # Two consecutive qualifying closes enter; the first false close exits.
        df["signal"] = (entry & entry.shift(1, fill_value=False)).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicator_data = pd.DataFrame({
        "coarse_share": [np.nan, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1],
        "upward_share": [1.0] * 8,
        "haar_8": [1.0] * 8,
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicator_data, None, WaveletParams())
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert result.data["size"].gt(0.0).all()
