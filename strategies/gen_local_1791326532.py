from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletCompressionParams:
    energy_window: int = 4
    baseline_window: int = 60
    compression_ratio: float = 0.35
    release_ratio: float = 2.5
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[WaveletCompressionParams]):
    strategy_id = "gen_local_1791326532"

    @classmethod
    def params_type(cls):
        return WaveletCompressionParams

    def warmup_bars(self, params: WaveletCompressionParams) -> int:
        return max(200, params.baseline_window + params.energy_window + 8)

    def indicators(self, data: pd.DataFrame, params: WaveletCompressionParams) -> pd.DataFrame:
        if min(params.energy_window, params.baseline_window, params.holding_bars) < 1:
            raise ValueError("Window lengths and holding_bars must be positive.")
        if not 0 < params.compression_ratio < 1 or params.release_ratio <= 1:
            raise ValueError("Require 0 < compression_ratio < 1 and release_ratio > 1.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        for half in (1, 2, 4):
            recent = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        out["coarse_energy"] = out[["energy_2", "energy_4"]].sum(axis=1, min_count=2)
        for name in ("total_energy", "coarse_energy"):
            out[f"{name}_baseline"] = out[name].rolling(
                params.baseline_window, min_periods=params.baseline_window
            ).mean().shift(1)
        out["range_width"] = (
            data["high"].rolling(8, min_periods=8).max()
            - data["low"].rolling(8, min_periods=8).min()
        ) / data["close"].where(data["close"] > 0)
        out["range_baseline"] = out["range_width"].rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean().shift(1)
        out["compressed"] = (
            (out["total_energy"] < params.compression_ratio * out["total_energy_baseline"])
            & (out["range_width"] < 0.75 * out["range_baseline"])
        )
        out["entry"] = (
            out["compressed"].shift(1, fill_value=False)
            & (out["haar_1"] > 0)
            & (out["haar_1"].pow(2) > params.release_ratio * out["energy_1"].shift(1))
            & (out["coarse_energy"] < 0.7 * out["coarse_energy_baseline"])
            & (data["close"] > out["ma200"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: WaveletCompressionParams
    ) -> SignalFrame:
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, enter in enumerate(indicators["entry"].fillna(False).to_numpy(dtype=bool)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif enter:
                signal[i] = 1
                exit_at = i + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    params = WaveletCompressionParams()
    index = pd.date_range("2020-01-01", periods=250, freq="B")
    close = pd.Series(np.linspace(100.0, 125.0, len(index)), index=index)
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000000.0
    }, index=index)
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["entry"].iloc[:199].any()
    forced = pd.DataFrame({"entry": True}, index=index)
    result = strategy.generate_signals(data, forced, None, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:9].eq(1).all()
    assert result["signal"].iloc[9] == 0
    assert result["signal"].iloc[10:18].eq(1).all()
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
