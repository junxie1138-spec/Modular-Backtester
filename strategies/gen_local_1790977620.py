from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapExhaustionParams:
    high_lookback: int = 63
    gap_fast: int = 3
    gap_slow: int = 9
    atr_period: int = 14
    drawdown_min: float = 0.03
    exhaustion_ratio: float = 0.65
    trail_atr: float = 3.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapExhaustionParams]):
    strategy_id = "gen_local_1790977620"

    @classmethod
    def params_type(cls):
        return GapExhaustionParams

    def warmup_bars(self, params):
        return int(max(params.high_lookback, params.atr_period + 1,
                       params.gap_fast + params.gap_slow + 1))

    def indicators(self, data, params):
        previous = data["close"].shift(1)
        gap = data["open"].div(previous.where(previous > 0)) - 1.0
        loss = (-gap).clip(lower=0.0)
        recent = loss.rolling(params.gap_fast, min_periods=params.gap_fast).mean()
        earlier = loss.shift(params.gap_fast).rolling(
            params.gap_slow, min_periods=params.gap_slow).mean()
        peak = data["close"].rolling(
            params.high_lookback, min_periods=params.high_lookback).max()
        depth = 1.0 - data["close"].div(peak.where(peak > 0))
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
        attenuation = (1.0 - recent.div(earlier.where(earlier > 0))).clip(0.0, 1.0)
        entry = (
            (depth >= params.drawdown_min)
            & (earlier > 0)
            & (recent <= params.exhaustion_ratio * earlier)
            & (gap < 0)
            & (data["close"] >= previous)
            & (atr > 0)
        )
        conviction = (
            0.65 * (depth / max(params.drawdown_min * 3.0, 1e-9)).clip(0.0, 1.0)
            + 0.35 * attenuation
        ).clip(0.0, 1.0)
        out = pd.DataFrame({
            "gap": gap,
            "gap_loss": loss,
            "recent_loss": recent,
            "earlier_loss": earlier,
            "peak": peak,
            "drawdown": depth,
            "true_range": tr,
            "atr": atr,
            "attenuation": attenuation,
            "conviction": conviction,
            "entry": entry.astype(int),
        }, index=data.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data, indicators, ctx, params):
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].to_numpy(dtype=bool)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        exposure = 1.0
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                age += 1
                if valid:
                    # Start tracking closes on the next-bar fill session.
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if not valid or close[i] < stop or age >= params.hold_bars:
                    active = False
                else:
                    signal[i] = 1
                    size[i] = exposure
            elif valid and entry[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                exposure = float(0.25 + 0.70 * np.clip(conviction[i], 0.0, 1.0))
                signal[i] = 1
                size[i] = exposure
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100, 101, 105, 104, 101, 100, 99]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 10, 10, 10, 10],
        "entry": [1, 0, 0, 0, 0, 0, 0],
        "conviction": [0.8] * 7,
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, GapExhaustionParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert abs(result["size"].iloc[1] - 0.81) < 1e-12
    indicators["atr"] = 1.0
    data["close"] = 100.0
    result = strategy.generate_signals(
        data, indicators, None, GapExhaustionParams(hold_bars=3)).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
