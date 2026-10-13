from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationPositionParams:
    roc_bars: int = 3
    channel_bars: int = 20
    percentile_bars: int = 60
    acceleration_quantile: float = 0.80
    position_quantile: float = 0.35
    atr_bars: int = 14
    trail_atr: float = 1.8
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[AccelerationPositionParams]):
    strategy_id = "gen_local-w2_1791248617"

    @classmethod
    def params_type(cls) -> type[AccelerationPositionParams]:
        return AccelerationPositionParams

    @staticmethod
    def warmup_bars(params: AccelerationPositionParams) -> int:
        return int(max(
            params.channel_bars + params.percentile_bars + 1,
            params.roc_bars + params.percentile_bars + 2,
            params.atr_bars + 1,
        ))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AccelerationPositionParams) -> pd.DataFrame:
        if min(params.roc_bars, params.channel_bars, params.percentile_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not (0.0 < params.acceleration_quantile < 1.0 and 0.0 < params.position_quantile < 1.0):
            raise ValueError("Quantiles must lie strictly between zero and one.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("ATR multiplier must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["roc"] = close.div(close.shift(params.roc_bars).replace(0.0, np.nan)).sub(1.0)
        out["acceleration"] = out["roc"].diff()
        out["channel_low"] = close.rolling(params.channel_bars, min_periods=params.channel_bars).min()
        out["channel_high"] = close.rolling(params.channel_bars, min_periods=params.channel_bars).max()
        out["channel_span"] = out["channel_high"] - out["channel_low"]
        out["position"] = (close - out["channel_low"]).div(out["channel_span"].replace(0.0, np.nan))
        out["position_change"] = out["position"].diff()
        out["acceleration_threshold"] = out["acceleration"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.acceleration_quantile).shift(1)
        out["position_threshold"] = out["position"].rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.position_quantile).shift(1)
        previous_close = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["acceleration"] > out["acceleration_threshold"])
            & (out["acceleration"].shift(1) <= out["acceleration_threshold"].shift(1))
            & (out["acceleration"] > 0.0)
            & (out["position"] <= out["position_threshold"])
            & (out["position_change"] > 0.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AccelerationPositionParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i, (price, volatility, entry) in enumerate(zip(closes, atr, entries)):
            valid = np.isfinite(price) and np.isfinite(volatility) and volatility > 0.0
            if active:
                age += 1
                if valid:
                    # First held close is on the bar after the entry decision.
                    high_water = price if np.isnan(high_water) else max(high_water, price)
                    stop = max(stop, high_water - params.trail_atr * volatility)
                if not valid or price <= stop or age >= params.hold_bars:
                    active = False
            elif valid and entry:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
            raw[i] = int(active)
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    close = np.array([100.0, 101.0, 105.0, 104.0, 102.0, 101.0])
    data = pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1000.0,
    }, index=index)
    derived = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
        "entry": [True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, AccelerationPositionParams(hold_bars=5))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    result = GeneratedStrategy.generate_signals(data, derived, None, AccelerationPositionParams(hold_bars=3))
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
