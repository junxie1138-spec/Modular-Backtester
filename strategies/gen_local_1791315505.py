from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RotationParams:
    anchor_window: int = 63
    volatility_window: int = 10
    atr_window: int = 14
    minimum_depth_atr: float = 4.0
    variance_threshold: float = 0.40
    trail_atr: float = 3.0
    max_hold_bars: int = 20


class GeneratedStrategy(BaseStrategy[RotationParams]):
    strategy_id = "gen_local_1791315505"

    @classmethod
    def params_type(cls) -> type[RotationParams]:
        return RotationParams

    def warmup_bars(self, params: RotationParams) -> int:
        return int(max(params.anchor_window, params.atr_window + 1, params.volatility_window + 3))

    def indicators(self, data: pd.DataFrame, params: RotationParams) -> pd.DataFrame:
        if min(params.anchor_window, params.volatility_window, params.atr_window, params.max_hold_bars) < 1:
            raise ValueError("Windows and holding period must be positive")
        if not (0.0 < params.variance_threshold < 0.5):
            raise ValueError("variance_threshold must lie between zero and one half")
        if not (np.isfinite(params.trail_atr) and params.trail_atr > 0.0 and np.isfinite(params.minimum_depth_atr) and params.minimum_depth_atr > 0.0):
            raise ValueError("ATR multipliers must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["return"] = close.div(previous.where(previous > 0.0)).sub(1.0)
        out["downside_variance"] = out["return"].clip(upper=0.0).pow(2).rolling(params.volatility_window).mean()
        out["upside_variance"] = out["return"].clip(lower=0.0).pow(2).rolling(params.volatility_window).mean()
        out["total_variance"] = out["downside_variance"] + out["upside_variance"]
        out["downside_share"] = out["downside_variance"].div(out["total_variance"].where(out["total_variance"] > 0.0))
        out["share_change"] = out["downside_share"].diff()
        out["anchor_high"] = close.rolling(params.anchor_window).max()
        out["anchor_low"] = close.rolling(params.anchor_window).min()
        out["anchor_span"] = out["anchor_high"] - out["anchor_low"]
        out["recovery_fraction"] = (close - out["anchor_low"]).div(out["anchor_span"].where(out["anchor_span"] > 0.0))
        out["depth_atr"] = out["anchor_span"].div(out["atr"].where(out["atr"] > 0.0))
        out["close_change"] = close.diff()
        out["repair_zone"] = (
            out["depth_atr"].ge(params.minimum_depth_atr)
            & out["recovery_fraction"].between(0.15, 0.65)
        )
        out["long_setup"] = (
            out["repair_zone"]
            & out["downside_share"].lt(params.variance_threshold)
            & out["share_change"].lt(0.0)
            & out["close_change"].gt(0.0)
        ).fillna(False)
        out["short_setup"] = (
            out["repair_zone"]
            & out["downside_share"].gt(1.0 - params.variance_threshold)
            & out["share_change"].gt(0.0)
            & out["close_change"].lt(0.0)
        ).fillna(False)
        out["long_entry"] = out["long_setup"] & out["long_setup"].shift(1, fill_value=False)
        out["short_entry"] = out["short_setup"] & out["short_setup"].shift(1, fill_value=False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RotationParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        high_water = low_water = stop = np.nan
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not (np.isfinite(close) and np.isfinite(atr) and atr > 0.0):
                position = 0
                continue
            if position:
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                if position == 1:
                    stop = max(stop, high_water - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    # Mirror the long high-water trail for short positions.
                    stop = min(stop, low_water + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or i - entry_bar >= params.max_hold_bars:
                    position = 0
                    continue
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                entry_bar = i
                high_water = low_water = close
                stop = close - position * params.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RotationParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100.0, 101.0, 104.0, 103.0, 102.0, 100.0, 100.0]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
        "long_entry": [False, True, False, False, False, False, False],
        "short_entry": False,
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]
    indicators["long_entry"] = False
    indicators["short_entry"] = [False, True, False, False, False, False, False]
    data["close"] = [100.0, 99.0, 96.0, 97.0, 98.0, 100.0, 100.0]
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
