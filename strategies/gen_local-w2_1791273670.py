from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletTransitionParams:
    energy_window: int = 5
    baseline_window: int = 60
    share_jump: float = 0.18
    flow_threshold: float = 0.12
    atr_window: int = 14
    atr_multiple: float = 4.0
    hold_bars: int = 18
    refractory_bars: int = 5


class GeneratedStrategy(BaseStrategy[WaveletTransitionParams]):
    strategy_id = "gen_local-w2_1791273670"

    @classmethod
    def params_type(cls) -> type[WaveletTransitionParams]:
        return WaveletTransitionParams

    @staticmethod
    def warmup_bars(params: WaveletTransitionParams) -> int:
        return int(max(16 + params.energy_window + params.baseline_window,
                       16 + params.energy_window + 10, params.atr_window + 1, 8))

    @staticmethod
    def indicators(data: pd.DataFrame, params: WaveletTransitionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        for scale in (2, 4, 8):
            mean = log_close.rolling(scale, min_periods=scale).mean()
            out[f"haar_{scale}"] = np.sqrt(scale / 2.0) * (mean - mean.shift(scale))
            out[f"energy_{scale}"] = out[f"haar_{scale}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window).mean()
        total = out[["energy_2", "energy_4", "energy_8"]].sum(axis=1, min_count=3)
        out["total_energy"] = total
        out["coarse_share"] = (out["energy_4"] + out["energy_8"]) / total.replace(0, np.nan)
        out["prior_share"] = out["coarse_share"].shift(1).rolling(10, min_periods=10).mean()
        out["prior_energy"] = total.shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window).mean()
        spread = (data["high"] - data["low"]).replace(0, np.nan)
        location = ((2 * data["close"] - data["high"] - data["low"]) / spread).clip(-1, 1).fillna(0)
        volume = data["volume"].clip(lower=0)
        out["volume_pressure"] = (location * volume).rolling(8, min_periods=8).sum() / volume.rolling(
            8, min_periods=8).sum().replace(0, np.nan)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        transition = (
            (out["coarse_share"] - out["prior_share"] >= params.share_jump)
            & (out["coarse_share"] >= 0.65)
            & (out["total_energy"] > 1.15 * out["prior_energy"])
        )
        # Trigger once per spectral transition, then require independent volume agreement.
        onset = transition & ~transition.shift(1, fill_value=False)
        long_entry = onset & (out["haar_8"] > 0) & (out["volume_pressure"] > params.flow_threshold)
        short_entry = onset & (out["haar_8"] < 0) & (out["volume_pressure"] < -params.flow_threshold)
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: WaveletTransitionParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = -np.inf
        low_water = np.inf
        trailing_floor = -np.inf
        short_ceiling = np.inf
        next_entry = 0
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                if valid:
                    high_water = max(high_water, close)
                    low_water = min(low_water, close)
                    trailing_floor = max(trailing_floor, high_water - params.atr_multiple * atr)
                    short_ceiling = min(short_ceiling, low_water + params.atr_multiple * atr)
                # The required highest-close drawdown exit applies to BOTH sides.
                # Shorts additionally exit on a ratcheting rebound from their lowest close.
                stopped = valid and (close < trailing_floor or (position == -1 and close > short_ceiling))
                if stopped or age >= params.hold_bars or not valid:
                    position = 0
                    next_entry = i + params.refractory_bars + 1
            elif valid and i >= next_entry and entries[i] != 0:
                position = int(entries[i])
                age = 0
                # Start tracking on the next bar, when the shifted entry fills.
                high_water, low_water = -np.inf, np.inf
                trailing_floor, short_ceiling = -np.inf, np.inf
            raw[i] = position
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=10)
    bars = pd.DataFrame({"close": [100, 102, 105, 106, 99, 100, 100, 100, 100, 100]}, index=index)
    derived = pd.DataFrame({"entry": [1, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                            "atr": [1, 1, 1, 1, 20, 1, 1, 1, 1, 1]}, index=index)
    result = GeneratedStrategy.generate_signals(
        bars, derived, None, WaveletTransitionParams(refractory_bars=1)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, -1, -1, -1]
    assert result["size"].gt(0).all()
