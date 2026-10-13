from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 24
    peak_window: int = 63
    history_window: int = 126
    bridge_percentile: float = 0.85
    drawdown_percentile: float = 0.60
    atr_window: int = 14
    breakeven_gain: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791255153"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.graph_window, params.peak_window) + params.history_window + params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.graph_window, params.peak_window, params.history_window, params.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < params.bridge_percentile < 1 and 0 < params.drawdown_percentile < 1):
            raise ValueError("Percentiles must lie strictly between zero and one.")
        if params.breakeven_gain <= 0 or params.trail_atr <= 0 or params.hold_bars < 1:
            raise ValueError("Exit parameters must be positive.")
        close = data["close"].astype(float).where(data["close"] > 0)
        log_close = np.log(close).replace([np.inf, -np.inf], np.nan)
        n, window = len(data), params.graph_window
        age = np.full(n, np.nan)
        if n >= window:
            blocks = np.lib.stride_tricks.sliding_window_view(log_close.to_numpy(), window)
            age[window - 1:] = np.where(
                np.isfinite(blocks).all(axis=1),
                np.argmin(blocks[:, ::-1], axis=1),
                np.nan,
            )
        trough_age = pd.Series(age, index=data.index)
        slope_max = pd.Series(-np.inf, index=data.index)
        bridges = pd.Series(0.0, index=data.index)
        # Natural visibility: each older neighbor must beat all intervening chord slopes.
        for lag in range(1, window):
            slope = (log_close.shift(lag) - log_close) / lag
            visible = slope.gt(slope_max) & slope.notna()
            bridges += (visible & trough_age.lt(lag)).astype(float)
            slope_max = pd.concat([slope_max, slope], axis=1).max(axis=1)
        capacity = (window - 1 - trough_age).clip(lower=1)
        bridge_share = bridges.div(capacity).where(trough_age.notna())
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        drawdown = 1.0 - close.div(peak)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(previous_close.notna())
        out = pd.DataFrame(index=data.index)
        out["trough_age"] = trough_age
        out["bridge_share"] = bridge_share
        out["bridge_threshold"] = bridge_share.rolling(
            params.history_window, min_periods=params.history_window
        ).quantile(params.bridge_percentile).shift(1)
        out["drawdown"] = drawdown
        out["drawdown_threshold"] = drawdown.rolling(
            params.history_window, min_periods=params.history_window
        ).quantile(params.drawdown_percentile).shift(1)
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["rising"] = close.diff().gt(0)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        ready = np.isfinite(indicators[[
            "trough_age", "bridge_share", "bridge_threshold",
            "drawdown", "drawdown_threshold", "atr",
        ]]).all(axis=1)
        candidates = (
            ready
            & indicators["trough_age"].between(3, params.graph_window // 2)
            & indicators["bridge_share"].gt(indicators["bridge_threshold"])
            & indicators["drawdown"].gt(indicators["drawdown_threshold"])
            & indicators["drawdown"].gt(0)
            & indicators["atr"].gt(0)
            & indicators["rising"]
        ).fillna(False).to_numpy(dtype=bool)
        candidates[:self.warmup_bars(params)] = False
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = stop = watermark = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]):
                    active = True
                    entry = opens[i]
                    entry_bar = i
                    watermark = entry
                    stop = entry - params.trail_atr * atr[i - 1]
                    armed = False
            if active:
                if not np.isfinite(closes[i]) or not np.isfinite(atr[i]):
                    active = False
                    continue
                watermark = max(watermark, closes[i])
                armed = armed or closes[i] >= entry * (1.0 + params.breakeven_gain)
                if armed:
                    stop = max(stop, entry, watermark - params.trail_atr * atr[i])
                # Stops are close-confirmed; the shifted signal exits at the next open.
                if closes[i] <= stop or i - entry_bar + 1 >= params.hold_bars:
                    active = False
                    continue
                decisions[i] = 1
            elif candidates[i]:
                pending = True
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params(graph_window=6, peak_window=2, history_window=2,
                    atr_window=2, breakeven_gain=0.01, trail_atr=2.0)
    strategy = object.__new__(GeneratedStrategy)
    count = strategy.warmup_bars(params) + 6
    index = pd.date_range("2020-01-01", periods=count)
    prices = np.full(count, 100.0)
    first = strategy.warmup_bars(params)
    prices[first + 1:first + 6] = [102.0, 104.0, 105.0, 102.0, 100.0]
    data = pd.DataFrame({
        "open": prices, "high": prices + 1, "low": prices - 1,
        "close": prices, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "trough_age": 3.0, "bridge_share": 0.0, "bridge_threshold": 0.5,
        "drawdown": 0.1, "drawdown_threshold": 0.05, "atr": 1.0,
        "rising": True,
    }, index=index)
    ind.loc[index[first], "bridge_share"] = 1.0
    ind.loc[index[first + 3:first + 5], "atr"] = 5.0
    result = strategy.generate_signals(data, ind, None, params).data
    expected = np.zeros(count, dtype=int)
    expected[first + 1:first + 5] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert (result["size"] > 0).all()
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:-2], params)
    pd.testing.assert_frame_equal(full.iloc[:-2], prefix)


if __name__ == "__main__":
    _self_check()
