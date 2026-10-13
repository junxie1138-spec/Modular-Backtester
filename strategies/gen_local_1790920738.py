from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    drawdown_window: int = 20
    roc_window: int = 3
    atr_window: int = 14
    min_depth_atr: float = 1.5
    acceleration_atr: float = 0.15
    trail_atr: float = 2.0
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1790920738"

    @classmethod
    def params_type(cls):
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(params.drawdown_window + 1, params.atr_window + 2, params.roc_window + 3))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        if min(params.drawdown_window, params.roc_window, params.atr_window) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five.")
        if min(params.min_depth_atr, params.acceleration_atr, params.trail_atr) <= 0:
            raise ValueError("Thresholds must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["peak"] = close.rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        out["roc"] = close.div(close.shift(params.roc_window)).sub(1)
        out["acceleration"] = out["roc"].diff()
        out["acceleration_scaled"] = out["acceleration"].div(out["atr"].div(close).where(out["atr"] > 0))
        out["depth_atr"] = (out["peak"] - close).div(out["atr"].where(out["atr"] > 0))
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out[["atr", "depth_atr", "roc", "acceleration_scaled"]].notna().all(axis=1)
        damaged = valid & (out["depth_atr"] >= params.min_depth_atr)
        out["long_setup"] = damaged & (out["roc"] < 0) & (out["acceleration_scaled"] > params.acceleration_atr)
        out["short_setup"] = damaged & (out["roc"] > 0) & (out["acceleration_scaled"] < -params.acceleration_atr)
        out["long_confirmed"] = out["long_setup"] & out["long_setup"].shift(1, fill_value=False)
        out["short_confirmed"] = out["short_setup"] & out["short_setup"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_confirmed"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if position:
                age += 1
                if not valid:
                    position = 0
                else:
                    # Entry fills on this bar after the final signal shift.
                    if age == 1:
                        extreme = close
                        stop = close - position * params.trail_atr * atr
                    if position == 1:
                        extreme = max(extreme, close)
                        stop = max(stop, extreme - params.trail_atr * atr)
                        stopped = close <= stop
                    else:
                        extreme = min(extreme, close)
                        stop = min(stop, extreme + params.trail_atr * atr)
                        stopped = close >= stop
                    if stopped or age >= params.holding_bars:
                        position = 0
            elif valid and (longs[i] or shorts[i]):
                position = 1 if longs[i] else -1
                age = 0
                extreme = np.nan
                stop = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    idx = pd.date_range("2020-01-01", periods=12)
    c = pd.Series([100, 100, 100, 102, 103, 99, 99, 99, 97, 96, 100, 100], index=idx, dtype=float)
    data = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000.0})
    params = RecoveryParams(holding_bars=5)
    strategy = GeneratedStrategy()
    derived = strategy.indicators(data, params)
    assert derived.index.equals(data.index)
    assert not derived["long_confirmed"].any()
    fixture = pd.DataFrame({
        "atr": [1.0] * 5 + [3.0] + [1.0] * 4 + [3.0, 1.0],
        "long_confirmed": [False, True] + [False] * 10,
        "short_confirmed": [False] * 7 + [True] + [False] * 4,
    }, index=idx)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, fixture, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, 0]
    assert (result["size"] > 0).all()
    recovering = data.iloc[:6].copy()
    recovering["close"] = [100.0, 90.0, 85.0, 83.0, 82.0, 81.5]
    recovering["open"] = recovering["close"]
    recovering["high"] = recovering["close"] + 1
    recovering["low"] = recovering["close"] - 1
    check_params = RecoveryParams(drawdown_window=3, roc_window=1, atr_window=2,
                                  min_depth_atr=0.1, acceleration_atr=0.01)
    confirmation = strategy.indicators(recovering, check_params)["long_confirmed"]
    assert confirmation.tolist() == [False, False, False, True, True, True]
