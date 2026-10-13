from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 20
    correlation_window: int = 60
    entry_score: float = 0.15
    hold_bars: int = 2
    min_size: float = 0.20
    max_size: float = 0.95
    sizing_scale: float = 0.50


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791204587"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        if params.energy_window < 2 or params.correlation_window < 3:
            raise ValueError("Rolling windows are too short.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2.")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")
        if not np.isfinite(params.entry_score) or params.entry_score < 0:
            raise ValueError("entry_score must be finite and nonnegative.")
        if not np.isfinite(params.sizing_scale) or params.sizing_scale <= 0:
            raise ValueError("sizing_scale must be finite and positive.")
        return int(max(params.energy_window, params.correlation_window) + 9)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        r = np.log(data["close"].where(data["close"] > 0)).diff()
        out["close_return"] = r
        total_energy = pd.Series(0.0, index=data.index)
        weighted_forecast = pd.Series(0.0, index=data.index)
        for scale in (2, 4, 8):
            half = scale // 2
            recent = r.rolling(half, min_periods=half).sum()
            coefficient = (recent - recent.shift(half)) / np.sqrt(scale)
            energy = coefficient.pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
            rho = coefficient.rolling(
                params.correlation_window, min_periods=params.correlation_window
            ).corr(coefficient.shift(1)).clip(-1.0, 1.0)
            standardized = coefficient / np.sqrt(energy.where(energy > 0))
            forecast = rho * standardized
            out[f"haar_{scale}"] = coefficient
            out[f"energy_{scale}"] = energy
            out[f"rho_{scale}"] = rho
            out[f"forecast_{scale}"] = forecast
            total_energy = total_energy + energy
            weighted_forecast = weighted_forecast + energy * forecast
        out["total_energy"] = total_energy
        out["fine_energy_share"] = out["energy_2"] / total_energy.where(total_energy > 0)
        out["score"] = weighted_forecast / total_energy.where(total_energy > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletParams
    ) -> SignalFrame:
        self.warmup_bars(params)
        score = indicators["score"]
        entry = (
            (indicators["close_return"] < 0)
            & (indicators["rho_2"] < 0)
            & (indicators["fine_energy_share"] > 0.5)
            & (score > params.entry_score)
            & np.isfinite(score)
        ).fillna(False).to_numpy(dtype=bool)
        strength = score.clip(lower=0).fillna(0)
        proposed_size = params.min_size + (params.max_size - params.min_size) * (
            strength / (strength + params.sizing_scale)
        )
        signal = np.zeros(len(data), dtype=int)
        size = np.full(len(data), params.min_size, dtype=float)
        exit_bar = -1
        held_size = params.min_size
        for i in range(len(data)):
            if i == exit_bar:
                # Emit a flat bar even if another entry qualifies on expiry.
                continue
            if i > exit_bar:
                if not entry[i]:
                    continue
                exit_bar = i + params.hold_bars
                held_size = float(proposed_size.iloc[i])
            signal[i] = 1
            size[i] = held_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"close": np.arange(100.0, 108.0)}, index=index)
    synthetic = pd.DataFrame({
        "close_return": -0.01, "rho_2": -0.5,
        "fine_energy_share": 0.75,
        "score": [0.2, 0.8, 0.1, 0.9, 0.3, 0.6, 0.4, 0.7],
    }, index=index)
    result = strategy.generate_signals(data, synthetic, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1]
    assert result["size"].iloc[1] == result["size"].iloc[2]
    assert result["size"].iloc[4] > result["size"].iloc[1]
    assert result["size"].between(params.min_size, params.max_size).all()
    flat = pd.DataFrame({"close": np.full(100, 100.0)},
                        index=pd.date_range("2020-01-01", periods=100))
    flat_indicators = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, flat_indicators, None, params).data["signal"].eq(0).all()
