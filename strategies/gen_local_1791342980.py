from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 24
    autocorr_window: int = 40
    min_energy_share: float = 0.40
    max_autocorr: float = -0.20
    stretch_threshold: float = 1.0
    hold_bars: int = 8

    def __post_init__(self):
        if self.energy_window < 2 or self.autocorr_window < 2 or self.hold_bars < 1:
            raise ValueError("Windows must exceed one; hold_bars must be positive.")
        if not 0 < self.min_energy_share < 1 or not -1 < self.max_autocorr < 0:
            raise ValueError("Invalid energy share or negative autocorrelation threshold.")
        if not np.isfinite(self.stretch_threshold) or self.stretch_threshold <= 0:
            raise ValueError("stretch_threshold must be finite and positive.")


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791342980"

    @classmethod
    def params_type(cls):
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(200, 16 + params.energy_window - 1,
                       12 + params.autocorr_window - 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        # Causal, undecimated Haar details: two adjacent blocks at each scale.
        for half in (2, 4, 8):
            block = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = (block - block.shift(half)) * np.sqrt(half / 2.0)
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["lagged_haar_4"] = out["haar_4"].shift(4)
        out["autocorr"] = out["haar_4"].rolling(
            params.autocorr_window, min_periods=params.autocorr_window
        ).corr(out["lagged_haar_4"]).clip(-1.0, 1.0)
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(
            axis=1, min_count=3
        )
        out["energy_share"] = out["energy_4"] / out["total_energy"].where(
            out["total_energy"] > 0
        )
        out["stretch"] = out["haar_4"] / np.sqrt(
            out["energy_4"].where(out["energy_4"] > 0)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        needed = ["ma200", "autocorr", "energy_share", "stretch"]
        ready = np.isfinite(indicators[needed]).all(axis=1)
        active = ready & (indicators["energy_share"] >= params.min_energy_share)
        active &= indicators["autocorr"] <= params.max_autocorr
        long_entry = active & (data["close"] > indicators["ma200"])
        long_entry &= indicators["stretch"] <= -params.stretch_threshold
        short_entry = active & (data["close"] < indicators["ma200"])
        short_entry &= indicators["stretch"] >= params.stretch_threshold
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position, exit_bar = 0, -1
        for bar, entry in enumerate(entries):
            if position:
                if bar == exit_bar:
                    position = 0
            elif entry:
                position = int(entry)
                exit_bar = bar + params.hold_bars
            signal[bar] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Check both directions, the one-bar delay, and expiry despite lost eligibility.
    strategy = GeneratedStrategy()
    params = WaveletParams()
    index = pd.date_range("2020-01-01", periods=12)
    for direction in (1, -1):
        data = pd.DataFrame({"close": 100.0 + direction}, index=index)
        derived = pd.DataFrame({
            "ma200": 100.0, "autocorr": -0.5,
            "energy_share": 0.0, "stretch": -2.0 * direction
        }, index=index)
        derived.iloc[0, derived.columns.get_loc("energy_share")] = 0.9
        actual = strategy.generate_signals(data, derived, None, params).data["signal"]
        expected = np.zeros(len(index), dtype=int)
        expected[1:params.hold_bars + 1] = direction
        assert np.array_equal(actual.to_numpy(), expected)
