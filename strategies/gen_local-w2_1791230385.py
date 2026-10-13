from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_share: float = 0.65
    stretch_threshold: float = 0.75
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791230385"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    @staticmethod
    def warmup_bars(params: WaveletParams) -> int:
        return 8

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if not 0.0 < params.energy_share < 1.0:
            raise ValueError("energy_share must be between zero and one")
        if params.stretch_threshold <= 0.0 or not np.isfinite(params.stretch_threshold):
            raise ValueError("stretch_threshold must be finite and positive")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        price = np.log(data["close"].where(data["close"] > 0.0))
        price = price.replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        for lag in range(8):
            out[f"log_lag_{lag}"] = price.shift(lag)
        fine_energy = pd.Series(0.0, index=data.index)
        for pair in range(4):
            detail = (out[f"log_lag_{2 * pair}"] - out[f"log_lag_{2 * pair + 1}"]) / np.sqrt(2.0)
            out[f"fine_detail_{pair}"] = detail
            fine_energy = fine_energy + detail.pow(2)
        recent = (out["log_lag_0"] + out["log_lag_1"] - out["log_lag_2"] - out["log_lag_3"]) / 2.0
        older = (out["log_lag_4"] + out["log_lag_5"] - out["log_lag_6"] - out["log_lag_7"]) / 2.0
        recent_sum = sum(out[f"log_lag_{lag}"] for lag in range(4))
        older_sum = sum(out[f"log_lag_{lag}"] for lag in range(4, 8))
        coarse = (recent_sum - older_sum) / np.sqrt(8.0)
        middle_energy = recent.pow(2) + older.pow(2)
        total_energy = fine_energy + middle_energy + coarse.pow(2)
        mean = price.rolling(8, min_periods=8).mean()
        scale = price.rolling(8, min_periods=8).std(ddof=0)
        out["middle_recent"] = recent
        out["middle_older"] = older
        out["coarse_detail"] = coarse
        out["fine_energy"] = fine_energy
        out["middle_energy"] = middle_energy
        out["total_energy"] = total_energy
        out["mean"] = mean
        out["scale"] = scale
        out["energy_share"] = middle_energy / total_energy.where(total_energy > 0.0)
        out["stretch"] = ((out["log_lag_0"] + out["log_lag_1"]) / 2.0 - mean) / scale.where(scale > 0.0)
        out["alternating"] = (recent * older < 0.0).astype(int)
        out["ready"] = (price.rolling(8, min_periods=8).count().eq(8) & total_energy.gt(0.0) & scale.gt(0.0)).astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        stretch = indicators["stretch"]
        qualified = (
            indicators["ready"].eq(1)
            & indicators["alternating"].eq(1)
            & indicators["energy_share"].ge(params.energy_share)
            & stretch.abs().ge(params.stretch_threshold)
            & (stretch * indicators["middle_recent"]).gt(0.0)
        )
        entries = np.where(qualified, -np.sign(stretch), 0).astype(int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        remaining = 0
        for bar, entry in enumerate(entries):
            if remaining > 0:
                remaining -= 1
                if remaining == 0:
                    position = 0
            elif entry != 0:
                position = int(entry)
                remaining = int(params.hold_bars)
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=24, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    flat = GeneratedStrategy.indicators(data, params)
    assert np.isfinite(flat.to_numpy()).all()
    assert GeneratedStrategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()
    for direction in (-1, 1):
        probe = flat.copy()
        probe.loc[index[0], ["ready", "alternating", "energy_share", "stretch", "middle_recent"]] = [1, 1, 0.9, -direction * 2.0, -direction]
        result = GeneratedStrategy.generate_signals(data, probe, None, params).data
        assert result["signal"].tolist() == [0] + [direction] * 18 + [0] * 5
        assert result["size"].gt(0.0).all()
