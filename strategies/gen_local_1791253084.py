from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletRecoveryParams:
    drawdown_lookback: int = 20
    min_drawdown: float = 0.015
    energy_floor: float = 0.12
    atr_period: int = 14
    atr_mult: float = 1.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WaveletRecoveryParams]):
    strategy_id = "gen_local_1791253084"

    @classmethod
    def params_type(cls):
        return WaveletRecoveryParams

    def warmup_bars(self, params: WaveletRecoveryParams) -> int:
        return int(max(params.drawdown_lookback, params.atr_period + 1, 11))

    def indicators(self, data: pd.DataFrame, params: WaveletRecoveryParams) -> pd.DataFrame:
        if params.drawdown_lookback < 2 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; drawdown lookback must be at least two.")
        if not (0.0 < params.min_drawdown < 1.0 and 0.0 < params.energy_floor < 1.0):
            raise ValueError("Drawdown and energy thresholds must lie strictly between zero and one.")
        if not np.isfinite(params.atr_mult) or params.atr_mult <= 0.0 or params.hold_bars not in (1, 2):
            raise ValueError("ATR multiplier must be positive and finite; holding period must be one or two bars.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0.0))
        for half in (1, 2, 4):
            mean = out["log_close"].rolling(half, min_periods=half).mean()
            out[f"mean_{half}"] = mean
            out[f"lagged_mean_{half}"] = mean.shift(half)
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (mean - out[f"lagged_mean_{half}"])
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(3, min_periods=3).mean()
        out["total_energy"] = out["energy_1"] + out["energy_2"] + out["energy_4"]
        out["fine_share"] = out["energy_1"] / out["total_energy"].where(out["total_energy"] > 0.0)
        out["previous_share"] = out["fine_share"].shift(1)
        out["previous_fast"] = out["haar_1"].shift(1)
        out["rolling_high"] = close.rolling(params.drawdown_lookback, min_periods=params.drawdown_lookback).max()
        out["drawdown"] = close / out["rolling_high"].where(out["rolling_high"] > 0.0) - 1.0
        out["previous_close"] = close.shift(1)
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["strength"] = ((out["fine_share"] - params.energy_floor) / (1.0 - params.energy_floor)).clip(0.0, 1.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out[["drawdown", "haar_1", "haar_2", "haar_4", "previous_fast", "fine_share", "previous_share", "atr", "strength"]].notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["drawdown"] <= -params.min_drawdown)
            & (out["haar_1"] > 0.0)
            & (out["previous_fast"] <= 0.0)
            & (out["haar_2"] < 0.0)
            & (out["haar_4"] < 0.0)
            & (out["fine_share"] > params.energy_floor)
            & (out["fine_share"] > out["previous_share"])
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletRecoveryParams) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        active = False
        age = 0
        highest_close = -np.inf
        stop = -np.inf
        trade_size = 0.25
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and close[i] > 0.0 and np.isfinite(atr[i]) and atr[i] > 0.0
            if active:
                age += 1
                if valid:
                    # Entry fills on this bar after the prior close's decision.
                    highest_close = max(highest_close, close[i])
                    stop = max(stop, highest_close - params.atr_mult * atr[i])
                if not valid or close[i] <= stop or age >= params.hold_bars:
                    active = False
                else:
                    signals[i] = 1
                    sizes[i] = trade_size
            elif valid and entry[i] and np.isfinite(strength[i]):
                active = True
                age = 0
                highest_close = -np.inf
                stop = -np.inf
                trade_size = float(0.25 + 0.75 * np.clip(strength[i], 0.0, 1.0))
                signals[i] = 1
                sizes[i] = trade_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    sample = pd.DataFrame({"close": [100.0, 102.0, 101.0, 100.0, 101.0]}, index=index)
    features = pd.DataFrame({"atr": 1.0, "strength": [0.8, 0.2, 0.2, 0.2, 0.2], "entry": [True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(sample, features, None, WaveletRecoveryParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert np.allclose(result.loc[index[1:3], "size"], 0.85)
    assert (result["size"] > 0.0).all()
    one_day = strategy.generate_signals(sample, features, None, WaveletRecoveryParams(hold_bars=1)).data
    assert one_day["signal"].tolist() == [0, 1, 0, 0, 0]
