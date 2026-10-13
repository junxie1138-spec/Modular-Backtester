from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 6
    baseline_window: int = 30
    breakout_window: int = 10
    atr_window: int = 14
    coarse_share: float = 0.65
    energy_multiple: float = 1.3
    trail_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1791310625"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(200, 8 + params.energy_window + params.baseline_window, params.breakout_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if min(params.energy_window, params.baseline_window, params.breakout_window, params.atr_window) < 1:
            raise ValueError("Lookback windows must be positive")
        if not 3 <= params.hold_days <= 5:
            raise ValueError("hold_days must be between three and five")
        if not 0 < params.coarse_share < 1 or not np.isfinite(params.coarse_share):
            raise ValueError("coarse_share must be between zero and one")
        if not np.isfinite(params.energy_multiple) or params.energy_multiple <= 0:
            raise ValueError("energy_multiple must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        close = data["close"].replace([np.inf, -np.inf], np.nan)
        out["log_close"] = np.log(close.where(close > 0))
        for width in (1, 2, 4):
            mean = out["log_close"].rolling(width, min_periods=width).mean()
            out[f"haar_{width}"] = np.sqrt(width / 2.0) * (mean - mean.shift(width))
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        out["coarse_fraction"] = out["energy_4"] / out["total_energy"].where(out["total_energy"] > 0)
        out["energy_baseline"] = out["total_energy"].shift(1).rolling(params.baseline_window, min_periods=params.baseline_window).mean()
        out["prior_channel_high"] = data["high"].shift(1).rolling(params.breakout_window, min_periods=params.breakout_window).max()
        out["ma_200"] = close.rolling(200, min_periods=200).mean()
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([(data["high"] - data["low"]), (data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["entry"] = (
            (close > out["ma_200"])
            & (close > out["prior_channel_high"])
            & (out["haar_4"] > 0)
            & (out["coarse_fraction"] >= params.coarse_share)
            & (out["coarse_fraction"] > out["coarse_fraction"].shift(1))
            & (out["energy_baseline"] > 0)
            & (out["total_energy"] > params.energy_multiple * out["energy_baseline"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma_200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(ma[i])
            if active:
                age += 1
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] <= stop or close[i] <= ma[i] or age >= params.hold_days:
                    active = False
            elif valid and entries[i] and close[i] > ma[i]:
                active = True
                age = 0
                # Entry fills next session; only subsequent closes enter the high-water mark.
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 110.0, 109.0, 107.0, 106.0]}, index=index)
    indicators = pd.DataFrame({"atr": [1.0, 1.0, 10.0, 10.0, 10.0], "ma_200": [90.0] * 5, "entry": [True, False, False, False, False]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, WaveletParams())
    # Widening ATR cannot lower the stop; the exit and entry both shift once.
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0]
    assert (result.data["size"] > 0).all()
