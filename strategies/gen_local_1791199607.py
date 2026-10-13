from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    channel_window: int = 20
    energy_window: int = 3
    decay_ratio: float = 0.6
    position_ceiling: float = 0.35
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791199607"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.channel_window, 8 + params.energy_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.channel_window < 2 or params.energy_window < 1:
            raise ValueError("Lookbacks must be positive; channel_window must be at least 2.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2.")
        if not 0.0 < params.decay_ratio < 1.0 or not 0.0 < params.position_ceiling < 1.0:
            raise ValueError("Thresholds must be strictly between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        for half in (1, 2, 4):
            recent = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = (recent - recent.shift(half)) * np.sqrt(half / 2.0)
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["prior_fine_energy"] = out["energy_1"].shift(2)
        out["range_low"] = data["low"].rolling(params.channel_window, min_periods=params.channel_window).min()
        out["range_high"] = data["high"].rolling(params.channel_window, min_periods=params.channel_window).max()
        out["range_width"] = (out["range_high"] - out["range_low"]).where(
            out["range_high"] > out["range_low"]
        )
        out["relative_position"] = (data["close"] - out["range_low"]) / out["range_width"]
        out["previous_close"] = data["close"].shift(1)
        out["wavelet_exhaustion"] = (
            (out["energy_1"] < params.decay_ratio * out["prior_fine_energy"])
            & (out["energy_4"] > out["energy_2"])
            & (out["haar_4"] < 0)
        )
        out["value_recovery"] = (
            out["relative_position"].between(0.0, params.position_ceiling)
            & (data["close"] > out["previous_close"])
            & (data["close"] > data["open"])
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2.")
        entries = (
            indicators["wavelet_exhaustion"].fillna(False)
            & indicators["value_recovery"].fillna(False)
        ).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Emit flat at expiry even if a new entry condition is true.
                    exit_bar = -1
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=32, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    features = GeneratedStrategy.indicators(None, data, params)
    flat = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert flat["signal"].eq(0).all()
    data = data.iloc[:8]
    features = pd.DataFrame({
        "wavelet_exhaustion": [True, False, True, True, False, False, False, False],
        "value_recovery": [True] * 8,
    }, index=data.index)
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 0]
    features["value_recovery"] = False
    assert GeneratedStrategy.generate_signals(None, data, features, None, params).data["signal"].eq(0).all()
    params.hold_bars = 1
    features.loc[:, "wavelet_exhaustion"] = True
    features.loc[:, "value_recovery"] = True
    result = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]


if __name__ == "__main__":
    _self_check()
