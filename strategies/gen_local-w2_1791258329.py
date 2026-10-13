from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 4
    fast_share_min: float = 0.30
    stretch_min: float = 0.65


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791258329"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return 8 + params.energy_window + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 1:
            raise ValueError("energy_window must be positive")
        if not np.isfinite(params.fast_share_min) or not 0 < params.fast_share_min < 1:
            raise ValueError("fast_share_min must be between zero and one")
        if not np.isfinite(params.stretch_min) or params.stretch_min <= 0:
            raise ValueError("stretch_min must be positive")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4):
            scale = 2 * half
            recent = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{scale}"] = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out["energy_2"] + out["energy_4"] + out["energy_8"]
        out["fast_share"] = out["energy_2"] / out["total_energy"].where(out["total_energy"] > 0)
        out["coarse_energy"] = out["energy_4"] + out["energy_8"]
        out["stretch"] = -(out["haar_4"] + out["haar_8"]) / np.sqrt(
            2.0 * out["coarse_energy"].where(out["coarse_energy"] > 0)
        )
        out["eligible"] = (
            out["fast_share"].ge(params.fast_share_min)
            & out["stretch"].ge(params.stretch_min)
            & np.isfinite(out["fast_share"])
            & np.isfinite(out["stretch"])
        )
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletParams,
    ) -> SignalFrame:
        eligible = indicators["eligible"].fillna(False).astype(bool)
        # Two consecutive eligible closes enter; the first false close exits.
        confirmed = eligible & eligible.shift(1, fill_value=False)
        df = pd.DataFrame(index=data.index)
        df["signal"] = confirmed.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame(
        {"eligible": [False, True, True, True, False, True, True, False, False]},
        index=index,
    )
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1, 0]
    assert (result["size"] > 0).all()
    flat = pd.DataFrame(
        {"close": 100.0}, index=pd.date_range("2020-01-01", periods=40)
    )
    assert not GeneratedStrategy.indicators(flat, params)["eligible"].any()


if __name__ == "__main__":
    _self_check()
