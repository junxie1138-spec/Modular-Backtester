from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownParams:
    peak_window: int = 20
    atr_window: int = 14
    min_depth_atr: float = 1.5
    max_depth_atr: float = 4.5
    deceleration_ratio: float = 0.55
    trail_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[DrawdownParams]):
    strategy_id = "gen_local-w2_1791243791"

    @classmethod
    def params_type(cls):
        return DrawdownParams

    def warmup_bars(self, params: DrawdownParams) -> int:
        return int(max(200, params.peak_window + 2, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: DrawdownParams) -> pd.DataFrame:
        close = data["close"]
        previous_close = close.shift(1)
        true_range = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = peak - close
        deepening = depth.diff()
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["atr"] = atr
        out["peak"] = peak
        out["depth"] = depth
        out["depth_atr"] = depth / atr.where(atr > 0)
        out["deepening"] = deepening
        out["previous_deepening"] = deepening.shift(1)
        out["stable_peak"] = peak.eq(peak.shift(1)) & peak.eq(peak.shift(2))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: DrawdownParams,
    ) -> SignalFrame:
        ready = indicators[
            ["ma200", "atr", "depth_atr", "deepening", "previous_deepening"]
        ].notna().all(axis=1)
        entry = (
            ready
            & (data["close"] > indicators["ma200"])
            & (indicators["atr"] > 0)
            & indicators["stable_peak"].fillna(False)
            & indicators["depth_atr"].between(params.min_depth_atr, params.max_depth_atr)
            & (indicators["deepening"] > 0)
            & (indicators["previous_deepening"] > 0)
            & (indicators["deepening"] <=
               params.deceleration_ratio * indicators["previous_deepening"])
        ).fillna(False).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = (
                np.isfinite(close[i]) and np.isfinite(atr[i])
                and atr[i] > 0 and np.isfinite(regime[i])
            )
            if active:
                if valid:
                    # Entry fills today if the previous close first requested a position.
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if (
                    not valid or close[i] <= stop or close[i] <= regime[i]
                    or i - entry_bar >= params.hold_days
                ):
                    active = False
            elif valid and entry[i]:
                active = True
                entry_bar = i
                # Start tracking closes on the next bar, when the entry actually fills.
                high_water = -np.inf
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    close = np.array([100.0, 102.0, 103.0, 100.5, 100.0, 100.0, 100.0, 100.0])
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"ma200": 90.0, "atr": [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
         "depth_atr": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         "deepening": 0.5, "previous_deepening": 2.0, "stable_peak": True},
        index=index,
    )
    strategy = GeneratedStrategy()
    params = DrawdownParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    indicators["atr"] = 1.0
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
