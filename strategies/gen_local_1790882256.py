from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    signature_window: int = 12
    breakout_window: int = 20
    area_threshold: float = 0.10
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790882256"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.signature_window, params.breakout_window) + 1

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.signature_window < 2 or params.breakout_window < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive; signature_window must be at least two.")
        if not 0.0 < params.area_threshold < 1.0:
            raise ValueError("area_threshold must be between zero and one.")
        if not 0.0 < params.min_size <= params.max_size <= 1.0:
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")

        out = pd.DataFrame(index=data.index)
        length = params.signature_window
        out["log_volume"] = np.log(data["volume"].where(data["volume"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["log_price"] = np.log(data["close"].where(data["close"] > 0)).replace([np.inf, -np.inf], np.nan)
        out["dv"] = out["log_volume"].diff()
        out["dp"] = out["log_price"].diff()
        out["cross_increment"] = (
            out["log_volume"].shift(1) * out["dp"]
            - out["log_price"].shift(1) * out["dv"]
        )
        out["cross_sum"] = out["cross_increment"].rolling(length).sum()
        out["volume_displacement"] = out["log_volume"].diff(length)
        out["price_displacement"] = out["log_price"].diff(length)
        # Translation correction resets each path to its own window origin.
        out["signature_area"] = 0.5 * (
            out["cross_sum"]
            - out["log_volume"].shift(length) * out["price_displacement"]
            + out["log_price"].shift(length) * out["volume_displacement"]
        )
        out["volume_variation"] = out["dv"].abs().rolling(length).sum()
        out["price_variation"] = out["dp"].abs().rolling(length).sum()
        out["area_scale"] = out["volume_variation"] * out["price_variation"]
        out["normalized_area"] = (
            2.0 * out["signature_area"] / out["area_scale"].where(out["area_scale"] > 0)
        ).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out["channel_high"] = data["high"].rolling(params.breakout_window).max().shift(1)
        out["eligible"] = (
            out["normalized_area"].ge(params.area_threshold)
            & data["close"].gt(out["channel_high"])
        ).fillna(False)
        out["entry_size"] = params.min_size + (
            params.max_size - params.min_size
        ) * (out["normalized_area"] / (3.0 * params.area_threshold)).clip(0.0, 1.0).fillna(0.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        eligible = indicators["eligible"].fillna(False).to_numpy(dtype=bool)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        deadline = -1
        held_size = params.min_size

        for i in range(len(data)):
            if deadline >= 0:
                if i < deadline:
                    signal[i] = 1
                    sizes[i] = held_size
                else:
                    # Emit an explicit flat bar; eligibility cannot cancel expiry.
                    deadline = -1
            elif eligible[i] and np.isfinite(entry_sizes[i]):
                deadline = i + params.hold_bars
                held_size = float(np.clip(entry_sizes[i], params.min_size, params.max_size))
                signal[i] = 1
                sizes[i] = held_size

        df = pd.DataFrame({"signal": signal, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams(signature_window=2, breakout_window=1)
    index = pd.date_range("2020-01-01", periods=45)
    data = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=index)
    path = data.iloc[:3].copy()
    path["volume"] = np.exp([0.0, 1.0, 1.0])
    path["close"] = np.exp([0.0, 0.0, 1.0])
    assert np.isclose(strategy.indicators(path, params)["normalized_area"].iloc[-1], 1.0)
    indicators = pd.DataFrame({"eligible": np.arange(len(data)) >= 2, "entry_size": np.linspace(0.3, 0.9, len(data))}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].iloc[:3].eq(0).all()
    assert result["signal"].iloc[3:3 + params.hold_bars].eq(1).all()
    assert result["signal"].iloc[3 + params.hold_bars] == 0
    assert result["size"].iloc[3:3 + params.hold_bars].nunique() == 1
    assert np.isclose(result["size"].iloc[3], indicators["entry_size"].iloc[2])
    assert result["size"].gt(0).all()
    assert set(result["signal"].unique()) <= {0, 1}


if __name__ == "__main__":
    _self_check()
