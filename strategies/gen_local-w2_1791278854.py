from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletCascadeParams:
    energy_window: int = 3
    baseline_window: int = 20
    cascade_min: float = 0.04
    energy_ratio: float = 1.0


class GeneratedStrategy(BaseStrategy[WaveletCascadeParams]):
    strategy_id = "gen_local-w2_1791278854"

    @classmethod
    def params_type(cls) -> type[WaveletCascadeParams]:
        return WaveletCascadeParams

    def warmup_bars(self, params: WaveletCascadeParams) -> int:
        return int(params.baseline_window + params.energy_window + 10)

    def indicators(self, data: pd.DataFrame, params: WaveletCascadeParams) -> pd.DataFrame:
        if params.energy_window < 1 or params.baseline_window < 2:
            raise ValueError("Energy window must be positive; baseline window must be at least two.")
        if not np.isfinite(params.cascade_min) or not 0 <= params.cascade_min <= 1:
            raise ValueError("Cascade threshold must be between zero and one.")
        if not np.isfinite(params.energy_ratio) or params.energy_ratio <= 0:
            raise ValueError("Energy ratio must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["return"] = np.log(data["close"].where(data["close"] > 0)).diff()
        for scale in (1, 2, 4):
            block = out["return"].rolling(scale, min_periods=scale).sum()
            out[f"haar_{scale}"] = (block - block.shift(scale)) / np.sqrt(2.0 * scale)
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out["energy_1"] + out["energy_2"] + out["energy_4"]
        out["fine_share"] = out["energy_1"] / out["total_energy"].where(out["total_energy"] > 0)
        out["cascade"] = out["fine_share"].diff()
        out["baseline_energy"] = out["total_energy"].rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean().shift(1)
        out["energy_expansion"] = out["total_energy"] / out["baseline_energy"].where(
            out["baseline_energy"] > 0
        )
        out["two_day_drift"] = out["return"].rolling(2, min_periods=2).sum()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["cascade"] > params.cascade_min)
            & (out["energy_expansion"] > params.energy_ratio)
            & (out["haar_2"] > 0)
            & (out["haar_4"] > 0)
            & (out["two_day_drift"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletCascadeParams,
    ) -> SignalFrame:
        entry = indicators["entry"].fillna(False).astype(bool)
        df = pd.DataFrame(index=data.index)
        # Two qualifying closes enter; the first false close exits.
        df["signal"] = (entry & entry.shift(1, fill_value=False)).astype(int)
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(index=index)
    derived = pd.DataFrame({"entry": [False, True, True, True, False, True, True, False]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, derived, None, WaveletCascadeParams())
    assert result.data["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
