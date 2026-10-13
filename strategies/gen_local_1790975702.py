from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EnergyPressureParams:
    energy_window: int = 8
    energy_fraction: float = 0.75
    energy_growth: float = 1.10
    pressure_window: int = 5
    pressure_threshold: float = 0.12
    atr_period: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[EnergyPressureParams]):
    strategy_id = "gen_local_1790975702"

    @classmethod
    def params_type(cls) -> type[EnergyPressureParams]:
        return EnergyPressureParams

    def warmup_bars(self, params: EnergyPressureParams) -> int:
        return int(max(16 + params.energy_window + 3,
                       params.atr_period + 1, params.pressure_window))

    def indicators(self, data: pd.DataFrame, params: EnergyPressureParams) -> pd.DataFrame:
        if min(params.energy_window, params.pressure_window, params.atr_period,
               params.hold_bars) < 1:
            raise ValueError("Window lengths and holding period must be positive")
        if not 0.0 < params.energy_fraction < 1.0:
            raise ValueError("energy_fraction must be between zero and one")
        if not np.isfinite(params.energy_growth) or params.energy_growth <= 0.0:
            raise ValueError("energy_growth must be positive and finite")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0.0:
            raise ValueError("stop_atr must be positive and finite")
        if not np.isfinite(params.pressure_threshold) or not 0.0 <= params.pressure_threshold <= 1.0:
            raise ValueError("pressure_threshold must be between zero and one")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0.0))
        for width in (2, 4, 8):
            mean = out["log_close"].rolling(width, min_periods=width).mean()
            out[f"haar_{width}"] = (mean - mean.shift(width)) * np.sqrt(width / 2.0)
            out[f"energy_{width}"] = out[f"haar_{width}"].pow(2).rolling(
                params.energy_window, min_periods=params.energy_window
            ).mean()
        out["slow_energy"] = out["energy_4"] + out["energy_8"]
        out["total_energy"] = out["slow_energy"] + out["energy_2"]
        out["slow_fraction"] = out["slow_energy"] / out["total_energy"].where(
            out["total_energy"] > 0.0
        )
        out["prior_slow_energy"] = out["slow_energy"].shift(3)
        out["wavelet_agrees"] = (
            (out["slow_fraction"] >= params.energy_fraction)
            & (out["slow_energy"] > params.energy_growth * out["prior_slow_energy"])
            & (out["haar_4"] > 0.0)
            & (out["haar_8"] > 0.0)
        ).fillna(False)
        out["candle_range"] = data["high"] - data["low"]
        out["body_pressure"] = ((data["close"] - data["open"]) / out["candle_range"].where(
            out["candle_range"] > 0.0
        )).clip(-1.0, 1.0)
        out.loc[out["candle_range"].eq(0.0), "body_pressure"] = 0.0
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0.0)
        out["signed_volume"] = out["body_pressure"] * out["valid_volume"]
        out["pressure_volume"] = out["valid_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum()
        out["pressure"] = out["signed_volume"].rolling(
            params.pressure_window, min_periods=params.pressure_window
        ).sum() / out["pressure_volume"].where(out["pressure_volume"] > 0.0)
        out["pressure_agrees"] = (out["pressure"] >= params.pressure_threshold).fillna(False)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["candle_range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs()
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: EnergyPressureParams) -> SignalFrame:
        eligible = (indicators["wavelet_agrees"].fillna(False)
                    & indicators["pressure_agrees"].fillna(False)).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending_entry = False
        entry_bar = -1
        fixed_stop = np.nan
        for i in range(len(data)):
            if pending_entry:
                pending_entry = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    entry_bar = i
                    fixed_stop = opens[i] - params.stop_atr * atr[i - 1]
            if active:
                stop_hit = np.isfinite(closes[i]) and closes[i] < fixed_stop
                expired = i - entry_bar + 1 >= params.hold_bars
                if stop_hit or expired:
                    active = False
                    continue
                decisions[i] = 1
            elif eligible[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                decisions[i] = 1
                pending_entry = True
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100.0] * 5,
                         "close": [100.0, 110.0, 98.0, 100.0, 100.0]}, index=index)
    features = pd.DataFrame({"wavelet_agrees": [True, False, False, False, False],
                             "pressure_agrees": [True] * 5, "atr": [1.0] * 5}, index=index)
    params = EnergyPressureParams(stop_atr=1.0)
    result = GeneratedStrategy.generate_signals(None, data, features, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    params.hold_bars = 1
    result = GeneratedStrategy.generate_signals(None, data, features, None, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 0, 0]
    assert result.data["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
