from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CompressionParams:
    energy_window: int = 6
    baseline_window: int = 48
    wick_window: int = 3
    compression_ratio: float = 0.75
    rejection_threshold: float = 0.12


class GeneratedStrategy(BaseStrategy[CompressionParams]):
    strategy_id = "gen_local-w2_1791266892"

    @classmethod
    def params_type(cls) -> type[CompressionParams]:
        return CompressionParams

    def warmup_bars(self, params: CompressionParams) -> int:
        return int(max(8 + 2 * params.energy_window + params.baseline_window, params.wick_window))

    def indicators(self, data: pd.DataFrame, params: CompressionParams) -> pd.DataFrame:
        if min(params.energy_window, params.baseline_window, params.wick_window) < 1:
            raise ValueError("Lookbacks must be positive")
        if not 0 < params.compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")
        if not 0 < params.rejection_threshold < 1:
            raise ValueError("rejection_threshold must be in (0, 1)")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4):
            out[f"mean_{half}"] = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = (out[f"mean_{half}"] - out[f"mean_{half}"].shift(half)) * np.sqrt(half / 2.0)
            out[f"squared_{half}"] = out[f"haar_{half}"] ** 2
            out[f"energy_{half}"] = out[f"squared_{half}"].rolling(params.energy_window, min_periods=params.energy_window).mean()
            out[f"baseline_{half}"] = out[f"energy_{half}"].shift(params.energy_window).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
            out[f"ratio_{half}"] = out[f"energy_{half}"] / out[f"baseline_{half}"].where(out[f"baseline_{half}"] > 0)
        out["compressed"] = (
            out["ratio_1"].le(params.compression_ratio)
            & out["ratio_2"].le(params.compression_ratio)
            & out["ratio_4"].le(params.compression_ratio)
        )
        out["displacement"] = out["haar_1"] + out["haar_2"] / np.sqrt(2.0) + out["haar_4"] / 2.0
        out["wavelet_vote"] = np.where(
            out["compressed"] & out["displacement"].lt(0), 1,
            np.where(out["compressed"] & out["displacement"].gt(0), -1, 0),
        )
        out["bar_range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["body_low"] = data[["open", "close"]].min(axis=1, skipna=False)
        out["body_high"] = data[["open", "close"]].max(axis=1, skipna=False)
        out["lower_wick"] = (out["body_low"] - data["low"]).clip(lower=0)
        out["upper_wick"] = (data["high"] - out["body_high"]).clip(lower=0)
        out["wick_balance"] = (out["lower_wick"] - out["upper_wick"]) / out["bar_range"]
        out["rejection"] = out["wick_balance"].rolling(params.wick_window, min_periods=params.wick_window).mean()
        out["wick_vote"] = np.where(
            out["rejection"].gt(params.rejection_threshold), 1,
            np.where(out["rejection"].lt(-params.rejection_threshold), -1, 0),
        )
        out["entry"] = np.where(
            out["wavelet_vote"].eq(out["wick_vote"]) & out["wavelet_vote"].ne(0),
            out["wavelet_vote"], 0,
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: CompressionParams) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, 0, -1, -1, 0, 1, 1, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, CompressionParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
