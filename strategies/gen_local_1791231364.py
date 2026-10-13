from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    pressure_window: int = 4
    volume_window: int = 20
    entry_threshold: float = 0.15
    volume_confirmation: float = 1.10
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791231364"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.volume_window + params.pressure_window + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.pressure_window < 2 or params.volume_window < 1:
            raise ValueError("Lookbacks must be positive; pressure_window must be at least two.")
        if not (params.entry_threshold > 0 and params.volume_confirmation > 0):
            raise ValueError("Signal thresholds must be positive.")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")
        out = pd.DataFrame(index=data.index)
        out["returns"] = data["close"].where(data["close"] > 0).pct_change(fill_method=None)
        volume = data["volume"].where(data["volume"] > 0)
        out["volume_baseline"] = volume.shift(1).rolling(params.volume_window).mean()
        out["relative_volume"] = volume / out["volume_baseline"]
        out["weighted_returns"] = out["returns"] * out["relative_volume"]
        window = params.pressure_window
        out["net_move"] = out["returns"].rolling(window).sum()
        out["gross_move"] = out["returns"].abs().rolling(window).sum()
        denominator = out["gross_move"].where(out["gross_move"] > 0)
        out["efficiency"] = (out["net_move"].abs() / denominator).clip(0, 1)
        out["volume_pressure"] = out["weighted_returns"].rolling(window).sum() / denominator
        out["volume_confirmation"] = out["relative_volume"].rolling(window).mean()
        out["recoil_score"] = -out["volume_pressure"] * (1 - out["efficiency"])
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        scores = indicators["recoil_score"].to_numpy(dtype=float)
        volumes = indicators["volume_confirmation"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        holding = False
        allocation = params.min_size
        for i, (score, volume) in enumerate(zip(scores, volumes)):
            confirmed = np.isfinite(score) and np.isfinite(volume) and volume >= params.volume_confirmation
            if confirmed:
                if holding and score <= -params.entry_threshold:
                    holding = False
                elif not holding and score >= params.entry_threshold:
                    holding = True
                    strength = float(np.clip(score / (2 * params.entry_threshold), 0, 1))
                    allocation = params.min_size + (params.max_size - params.min_size) * strength
            signals[i] = int(holding)
            sizes[i] = allocation
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    indicators = pd.DataFrame(
        {"recoil_score": [np.nan, 0.15, 0.0, -0.3, -0.30, 0.30, -0.30, 0.0],
         "volume_confirmation": [np.nan, 1.2, 1.2, 0.9, 1.2, 1.2, 1.2, 1.2]},
        index=index,
    )
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 1, 0]
    assert np.isclose(result["size"].iloc[2], 0.60)
    assert np.isclose(result["size"].iloc[6], 0.95)
    assert result["size"].gt(0).all()
    flat = pd.concat([data] * 5, ignore_index=True)
    flat.index = pd.date_range("2020-01-01", periods=len(flat))
    actual = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, actual, ctx, params).data["signal"].eq(0).all()
