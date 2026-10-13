from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveTransferParams:
    energy_smooth: int = 4
    migration_window: int = 6
    pressure_window: int = 3
    atr_window: int = 14
    energy_share: float = 0.60
    pressure_threshold: float = 0.15
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[WaveTransferParams]):
    strategy_id = "gen_local_1790899144"

    @classmethod
    def params_type(cls):
        return WaveTransferParams

    def warmup_bars(self, params):
        return int(max(
            16 + params.energy_smooth + params.migration_window + 1,
            2 * params.pressure_window,
            params.atr_window + 1,
        ))

    def indicators(self, data, params):
        if min(params.energy_smooth, params.migration_window,
               params.pressure_window, params.atr_window, params.hold_bars) < 1:
            raise ValueError("Window lengths and holding bars must be positive.")
        if not (0.5 < params.energy_share < 1.0
                and 0.0 <= params.pressure_threshold < 1.0
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid energy, pressure, or stop threshold.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["log_return"] = np.log(close.where(close > 0)).diff()
        for half in (2, 4, 8):
            # Causal, undecimated Haar details on adjacent return blocks.
            mean = out["log_return"].rolling(half).mean()
            out[f"return_mean_{half}"] = mean
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(
                params.energy_smooth
            ).mean()
        out["total_energy"] = out[["energy_2", "energy_4", "energy_8"]].sum(
            axis=1, min_count=3
        )
        denominator = out["total_energy"].where(out["total_energy"] > 0)
        out["fine_share"] = out["energy_2"] / denominator
        out["coarse_share"] = (out["energy_4"] + out["energy_8"]) / denominator
        out["prior_fine_peak"] = out["fine_share"].shift(1).rolling(
            params.migration_window
        ).max()
        out["wave_turn"] = (
            (out["coarse_share"] > params.energy_share)
            & (out["coarse_share"].shift(1) <= params.energy_share)
            & (out["prior_fine_peak"] > 0.5)
            & (out["haar_8"] > 0)
        ).fillna(False)
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        out["close_location"] = (
            (2.0 * close - data["high"] - data["low"]) / out["range"]
        ).clip(-1.0, 1.0).fillna(0.0)
        out["valid_volume"] = data["volume"].where(data["volume"] >= 0)
        out["pressure_flow"] = out["close_location"] * out["valid_volume"]
        out["pressure_volume"] = out["valid_volume"].rolling(
            params.pressure_window
        ).sum()
        out["pressure_sum"] = out["pressure_flow"].rolling(
            params.pressure_window
        ).sum()
        out["pressure"] = out["pressure_sum"] / out["pressure_volume"].where(
            out["pressure_volume"] > 0
        )
        out["previous_pressure"] = out["pressure"].shift(params.pressure_window)
        out["pressure_turn"] = (
            (out["pressure"] > params.pressure_threshold)
            & (out["previous_pressure"] < -params.pressure_threshold)
        ).fillna(False)
        out["previous_close"] = close.shift(1)
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["raw_range"] = data["high"] - data["low"]
        out["true_range"] = out[["raw_range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        entry = (
            indicators["wave_turn"].fillna(False)
            & indicators["pressure_turn"].fillna(False)
            & np.isfinite(indicators["atr"])
            & (indicators["atr"] > 0)
            & np.isfinite(data["close"])
        ).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        stop = np.nan
        for i in range(len(data)):
            if active:
                if age == 0:
                    # The prior close ordered entry; this bar's open is now known.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                age += 1
                if (not np.isfinite(stop) or not np.isfinite(closes[i])
                        or closes[i] < stop or age >= params.hold_bars):
                    active = False
                else:
                    signals[i] = 1
            elif entry[i]:
                active = True
                age = 0
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = WaveTransferParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 96.0,
        "close": [100.0, 101.0, 99.0, 97.0] + [100.0] * 8,
        "volume": 1000.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    forced = pd.DataFrame({
        "wave_turn": [True] + [False] * 4 + [True] + [False] * 6,
        "pressure_turn": [True] + [False] * 11,
        "atr": [1.0] + [100.0] * 11,
    }, index=index)
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1] + [0] * 8
    # Subsequent ATR changes cannot move the frozen stop.
    data["close"] = 100.0
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    assert (result["size"] > 0).all()
    computed = strategy.indicators(data, params)
    assert not computed["wave_turn"].any()
    assert not computed["pressure_turn"].any()
    assert strategy.generate_signals(data, computed, ctx, params).data["signal"].eq(0).all()
    assert computed.index.equals(data.index)


if __name__ == "__main__":
    _self_check()
