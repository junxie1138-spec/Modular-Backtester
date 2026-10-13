from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    frac_order: float = 0.6
    frac_window: int = 24
    percentile_window: int = 126
    tail_percentile: float = 0.15
    peak_window: int = 20
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790859456"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.frac_window + params.percentile_window + 1,
                       params.peak_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if not (0.0 < params.frac_order < 1.0
                and 0.0 < params.tail_percentile < 0.5
                and params.stop_atr > 0.0
                and params.hold_days in (1, 2)
                and min(params.frac_window, params.percentile_window,
                        params.peak_window, params.atr_window) >= 2):
            raise ValueError("Invalid recovery parameters")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        close = close.where(close > 0.0)
        previous = close.shift(1)
        out = pd.DataFrame(index=data.index)
        fractional = close.copy()
        weight = 1.0
        # ponytail: finite fractional memory; increase frac_window for longer memory.
        for lag in range(1, params.frac_window):
            weight *= (lag - 1.0 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["fractional_pressure"] = fractional / close
        out["tail_threshold"] = out["fractional_pressure"].rolling(
            params.percentile_window, min_periods=params.percentile_window
        ).quantile(params.tail_percentile).shift(1)
        out["peak"] = close.rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out["drawdown"] = close / out["peak"] - 1.0
        out["drawdown_recovery"] = out["drawdown"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        pressure = out["fractional_pressure"]
        threshold = out["tail_threshold"]
        out["setup"] = (
            (pressure.shift(1) <= threshold.shift(1))
            & (pressure > threshold)
            & (pressure > pressure.shift(1))
            & (out["drawdown"] < 0.0)
            & (out["drawdown_recovery"] > 0.0)
            & (close > previous)
            & (out["atr"] > 0.0)
            & np.isfinite(out["atr"])
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        setups = indicators["setup"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_bar = -1
        entry_atr = np.nan
        fixed_stop = np.nan
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    entry_bar = i
                    # Entry executes at this open; ATR was known at the prior close.
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                else:
                    active = False
            if active:
                if (not np.isfinite(closes[i]) or closes[i] <= 0.0
                        or closes[i] < fixed_stop
                        or i - entry_bar + 1 >= params.hold_days):
                    active = False
                else:
                    signals[i] = 1
            elif (setups[i] and np.isfinite(closes[i]) and closes[i] > 0.0
                  and np.isfinite(atrs[i]) and atrs[i] > 0.0):
                active = True
                pending = True
                entry_atr = atrs[i]
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 96.0,
        "close": 100.0, "volume": 1000.0
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0, "setup": [True, False, False, False]
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    data.loc[index[1], "close"] = 97.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    params.hold_days = 1
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
