from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_window: int = 12
    anchor_window: int = 36
    migration_window: int = 8
    innovation_threshold: float = 0.2
    migration_threshold: float = 0.02
    profit_target: float = 0.05
    max_hold: int = 18
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791270919"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        if min(params.trend_window, params.anchor_window) < 2:
            raise ValueError("Regression windows must be at least two bars")
        if params.migration_window < 1 or params.max_hold < 1:
            raise ValueError("Migration window and holding limit must be positive")
        if not all(np.isfinite(v) and v > 0 for v in (
            params.innovation_threshold, params.migration_threshold,
            params.profit_target, params.size
        )):
            raise ValueError("Thresholds, target and size must be positive and finite")
        return int(max(params.trend_window, params.anchor_window, params.migration_window) + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["log_midpoint"] = np.log(((data["high"] + data["low"]) / 2).where(
            (data["high"] + data["low"]) > 0
        ))
        out["midpoint_return"] = out["log_midpoint"].diff()
        out["return_vol"] = out["midpoint_return"].rolling(params.anchor_window).std(ddof=0)
        clock = pd.Series(np.arange(len(data), dtype=float), index=data.index)
        for name, window in (("slope", params.trend_window), ("anchor_slope", params.anchor_window)):
            denominator = window * (window * window - 1) / 12.0
            out[name] = (
                (clock * out["log_midpoint"]).rolling(window).sum()
                - clock.rolling(window).mean() * out["log_midpoint"].rolling(window).sum()
            ) / denominator
        out["prior_anchor_slope"] = out["anchor_slope"].shift(1)
        out["slope_innovation"] = (
            (out["slope"] - out["prior_anchor_slope"]) * np.sqrt(params.trend_window)
            / out["return_vol"].replace(0, np.nan)
        )
        out["up_void"] = (data["low"] - data["high"].shift(1)).clip(lower=0)
        out["down_void"] = (data["low"].shift(1) - data["high"]).clip(lower=0)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["void_balance"] = (out["up_void"] - out["down_void"]).rolling(
            params.migration_window
        ).sum()
        out["range_budget"] = out["true_range"].rolling(params.migration_window).sum()
        out["migration"] = out["void_balance"] / out["range_budget"].replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        long_setup = (
            (indicators["slope"] > 0)
            & (indicators["slope_innovation"] > params.innovation_threshold)
            & (indicators["migration"] > params.migration_threshold)
        )
        short_setup = (
            (indicators["slope"] < 0)
            & (indicators["slope_innovation"] < -params.innovation_threshold)
            & (indicators["migration"] < -params.migration_threshold)
        )
        setup = np.where(long_setup.fillna(False), 1, np.where(short_setup.fillna(False), -1, 0))
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if direction:
                if i == entry_bar:
                    # Only read the fill open when that bar has actually arrived.
                    entry_price = opens[i]
                gain = direction * (closes[i] / entry_price - 1.0) if entry_price > 0 else np.nan
                if (
                    not np.isfinite(entry_price) or entry_price <= 0
                    or gain >= params.profit_target
                    or i - entry_bar + 1 >= params.max_hold
                ):
                    direction = 0
                    signals[i] = 0
                    continue
            elif i >= warmup - 1 and setup[i] != 0:
                direction = int(setup[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = direction
        df = pd.DataFrame({"signal": signals, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams(max_hold=4)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    index = pd.date_range("2020-01-01", periods=50)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index
    )
    first = strategy.warmup_bars(params) - 1
    for direction in (1, -1):
        indicators = pd.DataFrame(
            {"slope": 0.0, "slope_innovation": 0.0, "migration": 0.0}, index=index
        )
        indicators.iloc[first] = [direction * 0.01, direction * 1.0, direction * 0.1]
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[first] == 0
        assert (result["signal"].iloc[first + 1:first + 5] == direction).all()
        assert result["signal"].iloc[first + 5] == 0
        targeted = data.copy()
        targeted.loc[index[first + 2], "close"] = 100.0 * (1 + direction * 0.06)
        result = strategy.generate_signals(targeted, indicators, ctx, params).data
        assert result["signal"].iloc[first + 2] == direction
        assert result["signal"].iloc[first + 3] == 0
        indicators.loc[index[first], "migration"] *= -1
        assert (strategy.generate_signals(data, indicators, ctx, params).data["signal"] == 0).all()
    actual = strategy.indicators(data, params)
    assert (strategy.generate_signals(data, actual, ctx, params).data["signal"] == 0).all()
    cutoff = first + 3
    prefix = strategy.generate_signals(
        targeted.iloc[:cutoff], strategy.indicators(targeted.iloc[:cutoff], params), ctx, params
    ).data
    full = strategy.generate_signals(targeted, strategy.indicators(targeted, params), ctx, params).data
    pd.testing.assert_frame_equal(prefix, full.iloc[:cutoff])


if __name__ == "__main__":
    _self_check()
