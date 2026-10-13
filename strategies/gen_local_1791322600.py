from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 16
    dominance: float = 0.55
    displacement: float = 0.001


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791322600"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return max(200, 16 + params.energy_window - 1)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 1 or not 0.0 < params.dominance < 1.0 or params.displacement < 0.0:
            raise ValueError("Invalid wavelet parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        for half in (2, 4, 8):
            recent = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"mean_{half}"] = recent
            out[f"haar_{half}"] = (recent - recent.shift(half)) * np.sqrt(half / 2.0)
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        denominator = out["total_energy"].where(out["total_energy"] > 0.0)
        out["coarse_share"] = out["energy_8"] / denominator
        out["fine_share"] = out["energy_2"] / denominator
        out["coarse_displacement"] = out["haar_8"] / 2.0
        out["fine_displacement"] = out["haar_2"]
        out["regime"] = np.select(
            [out["coarse_share"] >= params.dominance,
             out["fine_share"] >= params.dominance],
            [1, -1], default=0
        ).astype(int)
        trend = out["regime"].eq(1)
        recoil = out["regime"].eq(-1)
        long_wave = (
            (trend & out["coarse_displacement"].gt(params.displacement))
            | (recoil & out["fine_displacement"].lt(-params.displacement))
        )
        short_wave = (
            (trend & out["coarse_displacement"].lt(-params.displacement))
            | (recoil & out["fine_displacement"].gt(params.displacement))
        )
        ready = out[["ma200", "coarse_share", "fine_share"]].notna().all(axis=1)
        out["entry"] = np.select(
            [ready & long_wave & data["close"].gt(out["ma200"]),
             ready & short_wave & data["close"].lt(out["ma200"])],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=230, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index
    )
    indicators = GeneratedStrategy.indicators(None, data, params)
    assert indicators["entry"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(None, params) >= 200
    sample = data.iloc[:7]
    entries = pd.DataFrame({"entry": [1, 0, -1, -1, 0, 1, 1]}, index=sample.index)
    signals = GeneratedStrategy.generate_signals(None, sample, entries, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert signals["size"].gt(0).all()
