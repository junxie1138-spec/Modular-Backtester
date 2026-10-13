from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecoveryParams:
    range_window: int = 10
    peak_window: int = 40
    setup_window: int = 12
    atr_window: int = 14
    confirm_window: int = 5
    drawdown_min: float = 0.025
    concentration_low: float = 0.75
    concentration_high: float = 0.90
    trail_atr: float = 2.8
    hold_bars: int = 18

    def __post_init__(self):
        if min(self.range_window, self.peak_window, self.setup_window, self.atr_window, self.confirm_window, self.hold_bars) < 1:
            raise ValueError("Window lengths and holding period must be positive.")
        if not (0 < self.concentration_low < self.concentration_high <= 1):
            raise ValueError("Concentration thresholds must satisfy 0 < low < high <= 1.")
        if not (0 < self.drawdown_min < 1 and self.trail_atr > 0):
            raise ValueError("Drawdown and ATR multiplier must be positive.")


class GeneratedStrategy(BaseStrategy[RangeRecoveryParams]):
    strategy_id = "gen_local-w2_1791307895"

    @classmethod
    def params_type(cls):
        return RangeRecoveryParams

    def warmup_bars(self, params: RangeRecoveryParams) -> int:
        return int(max(200, params.peak_window + params.setup_window,
                       params.range_window + params.setup_window,
                       params.atr_window + 1, params.confirm_window))

    def indicators(self, data: pd.DataFrame, params: RangeRecoveryParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["daily_range"] = (data["high"] - data["low"]).clip(lower=0)
        out["range_sum"] = out["daily_range"].rolling(params.range_window).sum()
        out["range_square_sum"] = out["daily_range"].pow(2).rolling(params.range_window).sum()
        out["range_participation"] = out["range_sum"].pow(2).div(
            params.range_window * out["range_square_sum"].replace(0, np.nan)
        ).clip(0, 1)
        out["peak"] = data["close"].rolling(params.peak_window).max()
        out["drawdown"] = 1 - data["close"].div(out["peak"].replace(0, np.nan))
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            out["daily_range"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = data["close"].rolling(200).mean()
        out["recovery_mean"] = data["close"].rolling(params.confirm_window).mean()
        out["setup"] = (
            (out["drawdown"] >= params.drawdown_min)
            & (out["range_participation"] <= params.concentration_low)
        ).astype(int)
        out["recent_setup"] = out["setup"].rolling(params.setup_window).max()
        out["recovery_cross"] = (
            (out["range_participation"] >= params.concentration_high)
            & (out["range_participation"].shift(1) < params.concentration_high)
        )
        out["entry"] = (
            (out["recent_setup"] == 1)
            & out["recovery_cross"]
            & (data["close"] > out["recovery_mean"])
            & (data["close"] > out["ma200"])
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeRecoveryParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        df["signal"] = 0
        df["size"] = 0.95
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high_water = np.nan
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and np.isfinite(regime[i]) and atr[i] > 0
            if active:
                age += 1
                if not valid:
                    active = False
                else:
                    # Entry is filled on this bar after the preceding close's decision.
                    high_water = close[i] if np.isnan(high_water) else max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if close[i] <= stop or close[i] < regime[i] or age >= params.hold_bars:
                        active = False
                # An exit cannot reopen the trade on the same decision bar.
            elif valid and entries[i]:
                active = True
                high_water = np.nan
                stop = -np.inf
                age = 0
            signals[i] = int(active)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = RangeRecoveryParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=240)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    calculated = strategy.indicators(data, params)
    assert calculated.index.equals(data.index)
    assert not calculated["entry"].iloc[:199].any()
    data = data.iloc[:5].copy()
    data["close"] = [100.0, 104.0, 102.0, 101.0, 100.0]
    supplied = pd.DataFrame({"atr": [1.0, 1.0, 10.0, 1.0, 1.0],
                             "ma200": 90.0,
                             "entry": [True, False, False, False, False]}, index=data.index)
    result = strategy.generate_signals(data, supplied, None, params).data
    # The ATR spike must not loosen the stop; decisions must shift exactly once.
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
