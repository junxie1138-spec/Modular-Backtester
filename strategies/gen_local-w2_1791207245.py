from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CapacityParams:
    roc_bars: int = 5
    pressure_bars: int = 4
    atr_bars: int = 14
    roc_ceiling: float = -0.02
    acceleration_floor: float = 0.002
    impact_ratio: float = 0.7
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[CapacityParams]):
    strategy_id = "gen_local-w2_1791207245"

    @classmethod
    def params_type(cls):
        return CapacityParams

    def warmup_bars(self, params: CapacityParams) -> int:
        return int(max(params.roc_bars + 2, 2 * params.pressure_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: CapacityParams) -> pd.DataFrame:
        if min(params.roc_bars, params.pressure_bars, params.atr_bars, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding_bars must be positive.")
        if not (0 < params.impact_ratio < 1 and params.trail_atr > 0 and params.breakeven_pct > 0):
            raise ValueError("Invalid impact ratio or stop parameters.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(params.roc_bars, fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["previous_acceleration"] = out["acceleration"].shift(1)
        out["downside_return"] = (-out["return"]).clip(lower=0)
        out["down_volume"] = data["volume"].where(out["return"] < 0, 0.0).where(out["return"].notna())
        out["loss_sum"] = out["downside_return"].rolling(params.pressure_bars).sum()
        out["sell_volume"] = out["down_volume"].rolling(params.pressure_bars).sum()
        out["impact"] = out["loss_sum"] / out["sell_volume"].where(out["sell_volume"] > 0)
        out["previous_impact"] = out["impact"].shift(params.pressure_bars)
        out["previous_sell_volume"] = out["sell_volume"].shift(params.pressure_bars)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["acceleration_turn"] = (
            (out["roc"] < params.roc_ceiling)
            & (out["acceleration"] > params.acceleration_floor)
            & (out["previous_acceleration"] <= params.acceleration_floor)
        )
        out["capacity_absorption"] = (
            (out["impact"] < params.impact_ratio * out["previous_impact"])
            & (out["previous_impact"] > 0)
            & (out["sell_volume"] >= out["previous_sell_volume"])
        )
        out["entry"] = (
            out["acceleration_turn"] & out["capacity_absorption"]
            & (out["atr"] > 0) & close.notna()
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CapacityParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = pending = armed = False
        entry = peak = stop = 0.0
        age = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                    active, armed, age = True, False, 0
            if active:
                age += 1
                valid = np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0
                if valid:
                    peak = max(peak, closes[i])
                    armed = armed or closes[i] >= entry * (1 + params.breakeven_pct)
                    if armed:
                        stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # Close-observed stops become next-open exits via the mandatory shift.
                if (valid and closes[i] <= stop) or age >= params.holding_bars:
                    active = False
                signals[i] = int(active)
            elif entries[i]:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 99.0,
                         "close": [100, 100, 103, 105, 104, 102, 101, 101],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [False, True, False, False, False, False, False, False],
                               "atr": [1, 1, 1, 0.5, 10, 10, 1, 1]}, index=index)
    strategy = GeneratedStrategy()
    params = CapacityParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    params.holding_bars = 2
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
