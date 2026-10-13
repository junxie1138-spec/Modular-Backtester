from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 5
    atr_window: int = 14
    min_strength: float = 0.12
    min_participation: float = 2.2
    min_coherence: float = 0.85
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local-w2_1791273307"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(16 + params.energy_window - 1, 17, params.atr_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        if params.energy_window < 1 or params.atr_window < 1 or params.max_hold < 1:
            raise ValueError("Window lengths and max_hold must be positive")
        if not (0 < params.min_strength <= 1 and 1 <= params.min_participation <= 3):
            raise ValueError("Invalid energy thresholds")
        if not (0 < params.min_coherence <= 1 and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Invalid coherence or trailing distance")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(lambda x: x > 0)
        out["log_close"] = np.log(close)
        out["log_return"] = out["log_close"].diff()
        out["return_power"] = out["log_return"].pow(2).rolling(16, min_periods=16).mean()
        for half in (2, 4, 8):
            recent = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (recent - recent.shift(half))
            # Normalize Haar coefficients into comparable slope units.
            out[f"slope_{half}"] = out[f"haar_{half}"] * np.sqrt(2.0) / half ** 1.5
            out[f"energy_{half}"] = out[f"slope_{half}"].pow(2).rolling(params.energy_window, min_periods=params.energy_window).mean()
            out[f"signed_energy_{half}"] = (np.sign(out[f"slope_{half}"]) * out[f"slope_{half}"].pow(2)).rolling(params.energy_window, min_periods=params.energy_window).mean()
        energies = out[[f"energy_{half}" for half in (2, 4, 8)]]
        out["total_energy"] = energies.sum(axis=1, min_count=3)
        out["energy_concentration"] = energies.pow(2).sum(axis=1, min_count=3)
        out["participation"] = out["total_energy"].pow(2) / out["energy_concentration"].where(out["energy_concentration"] > 0)
        out["strength"] = out["total_energy"] / (3.0 * out["return_power"].where(out["return_power"] > 0))
        out["coherence"] = out[[f"signed_energy_{half}" for half in (2, 4, 8)]].sum(axis=1, min_count=3) / out["total_energy"].where(out["total_energy"] > 0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([data["high"] - data["low"], (data["high"] - previous).abs(), (data["low"] - previous).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        required = indicators[["strength", "participation", "coherence", "atr"]]
        valid = pd.Series(np.isfinite(required.to_numpy()).all(axis=1), index=data.index)
        valid &= np.isfinite(data["close"]) & (data["close"] > 0) & (indicators["atr"] > 0)
        qualified = valid & (indicators["strength"] >= params.min_strength) & (indicators["participation"] >= params.min_participation)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[qualified & (indicators["coherence"] >= params.min_coherence)] = 1
        candidate.loc[qualified & (indicators["coherence"] <= -params.min_coherence)] = -1
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        confirmations = 0
        age = 0
        high_water = np.nan
        low_water = np.nan
        stop = np.nan
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = candidate.to_numpy(dtype=int)
        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            if position:
                age += 1
                exit_now = age >= params.max_hold
                if np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0:
                    if position == 1:
                        high_water = close if age == 1 else max(high_water, close)
                        proposed = high_water - params.trail_atr * atr
                        stop = proposed if age == 1 else max(stop, proposed)
                        exit_now |= close <= stop
                    else:
                        # Short-side mirror: the lowest close and stop only ratchet down.
                        low_water = close if age == 1 else min(low_water, close)
                        proposed = low_water + params.trail_atr * atr
                        stop = proposed if age == 1 else min(stop, proposed)
                        exit_now |= close >= stop
                else:
                    exit_now = True
                if exit_now:
                    position = 0
                    pending = 0
                    confirmations = 0
                    age = 0
                    stop = np.nan
            else:
                direction = int(candidates[i])
                confirmations = confirmations + 1 if direction != 0 and direction == pending else int(direction != 0)
                pending = direction
                if confirmations >= 2:
                    position = direction
                    age = 0
                    high_water = np.nan
                    low_water = np.nan
                    stop = np.nan
                    pending = 0
                    confirmations = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = WaveletParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100.0, 101.0, 102.0, 105.0, 104.0, 102.0, 101.0]}, index=index)
    indicators = pd.DataFrame({"strength": 0.5, "participation": 3.0, "coherence": 1.0, "atr": [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]
    indicators["coherence"] = -1.0
    data["close"] = 202.0 - data["close"]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, -1, -1, -1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
