from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockRepairParams:
    range_window: int = 20
    atr_window: int = 14
    recovery_window: int = 6
    shock_multiple: float = 1.8
    depth_atr: float = 1.5
    contraction: float = 0.65
    trail_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[ShockRepairParams]):
    strategy_id = "gen_local-w2_1791342077"

    @classmethod
    def params_type(cls):
        return ShockRepairParams

    def warmup_bars(self, params: ShockRepairParams) -> int:
        return int(max(200, max(params.range_window, params.atr_window) + params.recovery_window + 2))

    def indicators(self, data: pd.DataFrame, params: ShockRepairParams) -> pd.DataFrame:
        if min(params.range_window, params.atr_window, params.holding_bars) < 1 or params.recovery_window < 2:
            raise ValueError("Windows must be positive; recovery_window must be at least two.")
        if not all(np.isfinite(x) and x > 0 for x in (params.shock_multiple, params.depth_atr, params.trail_atr)):
            raise ValueError("Shock, depth, and trailing multipliers must be finite and positive.")
        if not 0 < params.contraction < 1:
            raise ValueError("contraction must lie strictly between zero and one.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["range"] = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["baseline_range"] = out["range"].rolling(params.range_window, min_periods=params.range_window).median().shift(1)
        out["previous_peak"] = close.rolling(params.range_window, min_periods=params.range_window).max().shift(1)
        out["previous_trough"] = close.rolling(params.range_window, min_periods=params.range_window).min().shift(1)
        out["close_location"] = (close - data["low"]) / out["range"].replace(0.0, np.nan)
        out["drawdown_atr"] = (out["previous_peak"] - close) / out["atr"].replace(0.0, np.nan)
        out["rally_atr"] = (close - out["previous_trough"]) / out["atr"].replace(0.0, np.nan)
        out["wide_range"] = (out["baseline_range"] > 0) & (out["range"] >= params.shock_multiple * out["baseline_range"])
        out["down_shock"] = out["wide_range"] & (out["drawdown_atr"] >= params.depth_atr) & (out["close_location"] <= 0.25) & (close < out["previous_close"])
        out["up_shock"] = out["wide_range"] & (out["rally_atr"] >= params.depth_atr) & (out["close_location"] >= 0.75) & (close > out["previous_close"])
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["bar_number"] = np.arange(len(data), dtype=float)
        for side in ("down", "up"):
            shock = out[side + "_shock"]
            out[side + "_midpoint"] = out["midpoint"].where(shock).ffill(limit=params.recovery_window)
            out[side + "_range"] = out["range"].where(shock).ffill(limit=params.recovery_window)
            out[side + "_age"] = out["bar_number"] - out["bar_number"].where(shock).ffill(limit=params.recovery_window)
        out["long_entry"] = (
            (close > out["ma200"]) & (out["down_age"] >= 2) & (out["down_age"] <= params.recovery_window)
            & (out["range"] <= params.contraction * out["down_range"])
            & (out["previous_close"] <= out["down_midpoint"]) & (close > out["down_midpoint"])
        ).fillna(False)
        out["short_entry"] = (
            (close < out["ma200"]) & (out["up_age"] >= 2) & (out["up_age"] <= params.recovery_window)
            & (out["range"] <= params.contraction * out["up_range"])
            & (out["previous_close"] >= out["up_midpoint"]) & (close < out["up_midpoint"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: ShockRepairParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        watermark = np.nan
        stop = np.nan
        rows = zip(data["close"].to_numpy(), indicators["atr"].to_numpy(),
                   indicators["ma200"].to_numpy(), indicators["long_entry"].fillna(False).to_numpy(),
                   indicators["short_entry"].fillna(False).to_numpy())
        for i, (close, atr, regime, enter_long, enter_short) in enumerate(rows):
            if not (np.isfinite(close) and np.isfinite(atr) and np.isfinite(regime) and close > 0 and atr > 0):
                position = 0
                age = 0
                watermark = stop = np.nan
                continue
            if position:
                # The first held close is on the bar after the entry decision.
                age += 1
                if position == 1:
                    watermark = close if np.isnan(watermark) else max(watermark, close)
                    candidate = watermark - params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else max(stop, candidate)
                    exit_now = close <= stop or close <= regime
                else:
                    watermark = close if np.isnan(watermark) else min(watermark, close)
                    candidate = watermark + params.trail_atr * atr
                    stop = candidate if np.isnan(stop) else min(stop, candidate)
                    exit_now = close >= stop or close >= regime
                if exit_now or age >= params.holding_bars:
                    position = 0
                    age = 0
                    watermark = stop = np.nan
                raw[i] = position
                continue
            if enter_long or enter_short:
                position = 1 if enter_long else -1
                age = 0
                watermark = stop = np.nan
            raw[i] = position
        df["signal"] = raw
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=11, freq="D")
    data = pd.DataFrame({"close": [100., 102., 105., 104., 102., 100., 98., 95., 96., 98., 99.]}, index=index)
    ind = pd.DataFrame({
        "atr": [1., 1., 1., 5., 5., 1., 1., 1., 5., 5., 1.],
        "ma200": [90.] * 5 + [110.] * 6,
        "long_entry": [True] + [False] * 10,
        "short_entry": [False] * 5 + [True] + [False] * 5,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, ind, ctx, ShockRepairParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert (result["size"] > 0).all()
    result = strategy.generate_signals(data.iloc[:3], ind.iloc[:3], ctx, ShockRepairParams()).data
    assert result["signal"].tolist() == [0, 1, 1]
    result = strategy.generate_signals(data.iloc[:4], ind.iloc[:4], ctx, ShockRepairParams(holding_bars=2)).data
    assert result["signal"].tolist() == [0, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
