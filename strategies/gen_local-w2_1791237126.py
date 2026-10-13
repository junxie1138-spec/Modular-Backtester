from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownShelfParams:
    peak_window: int = 63
    atr_window: int = 14
    shelf_window: int = 4
    depth_min: float = 2.0
    depth_max: float = 6.0
    shelf_width: float = 0.8
    reset_depth: float = 0.75
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[DrawdownShelfParams]):
    strategy_id = "gen_local-w2_1791237126"

    @classmethod
    def params_type(cls) -> type[DrawdownShelfParams]:
        return DrawdownShelfParams

    def warmup_bars(self, params: DrawdownShelfParams) -> int:
        if min(params.peak_window, params.atr_window, params.shelf_window, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not (0 <= params.reset_depth < params.depth_min < params.depth_max):
            raise ValueError("Require reset_depth < depth_min < depth_max")
        if params.stop_atr <= 0 or params.shelf_width <= 0:
            raise ValueError("Stop multiplier and shelf width must be positive")
        return int(max(params.peak_window, params.atr_window + 1, params.shelf_window) + 1)

    def indicators(self, data: pd.DataFrame, params: DrawdownShelfParams) -> pd.DataFrame:
        warmup = self.warmup_bars(params)
        close = data["close"]
        previous_close = close.shift(1)
        out = pd.DataFrame(index=data.index)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["peak"] = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        out["shelf_high"] = close.rolling(params.shelf_window, min_periods=params.shelf_window).max()
        out["shelf_low"] = close.rolling(params.shelf_window, min_periods=params.shelf_window).min()
        denominator = out["atr"].where(out["atr"] > 0)
        out["depth"] = ((out["peak"] - close) / denominator).replace([np.inf, -np.inf], np.nan)
        out["shelf_width"] = ((out["shelf_high"] - out["shelf_low"]) / denominator).replace([np.inf, -np.inf], np.nan)
        out["setup"] = (
            out["depth"].between(params.depth_min, params.depth_max)
            & out["shelf_width"].le(params.shelf_width)
            & close.gt(0)
        ).fillna(False)
        out["confirmed"] = out["setup"] & out["setup"].shift(1, fill_value=False)
        out["ready"] = np.arange(len(data)) >= warmup - 1
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DrawdownShelfParams) -> SignalFrame:
        self.warmup_bars(params)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        depth = indicators["depth"].to_numpy(dtype=float)
        confirmed = (indicators["confirmed"] & indicators["ready"]).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        armed = True
        entry_bar = -1
        stop_price = np.nan

        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # The prior close placed the order; anchor the fixed stop at its next-open fill.
                    if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                        active = False
                        continue
                    stop_price = opens[i] - params.stop_atr * atr[i - 1]
                if not np.isfinite(closes[i]) or closes[i] < stop_price or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                else:
                    signal[i] = 1
                continue

            if np.isfinite(depth[i]) and depth[i] <= params.reset_depth:
                armed = True
            if armed and confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                armed = False
                entry_bar = i + 1
                signal[i] = 1

        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = DrawdownShelfParams()
    close = np.concatenate([np.full(64, 100.0), np.full(26, 95.0)])
    data = pd.DataFrame({
        "open": close.copy(), "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=len(close)))
    indicators = strategy.indicators(data, params)
    assert not bool(indicators["confirmed"].iloc[67])
    assert bool(indicators["confirmed"].iloc[68])
    result = strategy.generate_signals(data, indicators, None, params).data
    expected = np.zeros(len(data), dtype=int)
    expected[69:77] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
    gap = data.copy()
    gap.iloc[69, gap.columns.get_loc("open")] = 100.0
    gap.iloc[69, gap.columns.get_loc("high")] = 100.5
    stopped = strategy.generate_signals(gap, strategy.indicators(gap, params), None, params).data
    expected[:] = 0
    expected[69] = 1
    assert np.array_equal(stopped["signal"].to_numpy(), expected)


if __name__ == "__main__":
    _self_check()
