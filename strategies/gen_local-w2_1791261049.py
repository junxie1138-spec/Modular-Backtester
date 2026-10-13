from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletPositionParams:
    position_cutoff: float = 0.30
    energy_ratio: float = 1.5


class GeneratedStrategy(BaseStrategy[WaveletPositionParams]):
    strategy_id = "gen_local-w2_1791261049"

    @classmethod
    def params_type(cls) -> type[WaveletPositionParams]:
        return WaveletPositionParams

    @staticmethod
    def warmup_bars(params: WaveletPositionParams) -> int:
        return 8

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletPositionParams) -> pd.DataFrame:
        if not 0.0 < params.position_cutoff < 0.5 or params.energy_ratio <= 0.0:
            raise ValueError("Require 0 < position_cutoff < 0.5 and energy_ratio > 0.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        for half in (1, 2, 4):
            width = 2 * half
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"mean_{half}"] = mean
            out[f"detail_{width}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{width}"] = out[f"detail_{width}"].pow(2)
            out[f"low_{width}"] = data["low"].rolling(width, min_periods=width).min()
            out[f"high_{width}"] = data["high"].rolling(width, min_periods=width).max()
            span = out[f"high_{width}"] - out[f"low_{width}"]
            out[f"position_{width}"] = (
                (data["close"] - out[f"low_{width}"]) / span.where(span > 0.0)
            ).clip(0.0, 1.0)
        out["total_energy"] = out["energy_2"] + out["energy_4"] + out["energy_8"]
        out["relative_position"] = (
            out["energy_2"] * out["position_2"]
            + out["energy_4"] * out["position_4"]
            + out["energy_8"] * out["position_8"]
        ) / out["total_energy"].where(out["total_energy"] > 0.0)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: WaveletPositionParams,
    ) -> SignalFrame:
        ready = indicators[[
            "detail_2", "detail_8", "energy_2", "energy_4",
            "energy_8", "relative_position",
        ]].notna().all(axis=1)
        entry = (
            ready
            & (indicators["detail_2"] < 0.0)
            & (indicators["detail_8"] > 0.0)
            & (indicators["relative_position"] < params.position_cutoff)
            & (
                indicators["energy_2"]
                > params.energy_ratio * (indicators["energy_4"] + indicators["energy_8"])
            )
        )
        df = pd.DataFrame(index=data.index)
        # The target becomes flat only when the complete entry condition flips false.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = WaveletPositionParams()
    close = np.array([101, 102, 102, 103, 104, 105, 106, 100, 103, 104, 99, 101], dtype=float)
    data = pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1,
         "close": close, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=len(close)),
    )
    indicators = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert signals.index.equals(data.index)
    assert set(signals["signal"].unique()) <= {0, 1}
    assert (signals["size"] > 0.0).all()
    assert signals["signal"].iloc[8] == 1
    assert signals["signal"].iloc[9] == 0
    for end in range(1, len(data) + 1):
        prefix = data.iloc[:end]
        prefix_indicators = GeneratedStrategy.indicators(prefix, params)
        prefix_signals = GeneratedStrategy.generate_signals(prefix, prefix_indicators, None, params).data
        pd.testing.assert_frame_equal(prefix_indicators, indicators.iloc[:end])
        pd.testing.assert_frame_equal(prefix_signals, signals.iloc[:end])
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    flat_signals = GeneratedStrategy.generate_signals(
        flat, GeneratedStrategy.indicators(flat, params), None, params
    ).data
    assert (flat_signals["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
