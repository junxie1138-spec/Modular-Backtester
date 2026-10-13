from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fd_window: int = 16
    frac_order: float = 0.65
    scale_window: int = 12
    body_window: int = 2
    impulse_threshold: float = 0.15
    body_threshold: float = 0.12

    def __post_init__(self):
        if min(self.fd_window, self.scale_window, self.body_window) < 2:
            raise ValueError("Windows must be at least two bars.")
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError("frac_order must be between zero and one.")
        if not np.isfinite(self.impulse_threshold) or self.impulse_threshold <= 0:
            raise ValueError("impulse_threshold must be finite and positive.")
        if not 0.0 < self.body_threshold < 1.0:
            raise ValueError("body_threshold must be between zero and one.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790869469"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fd_window + 1, params.scale_window + 1, params.body_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        # ponytail: fixed-width fractional memory; enlarge fd_window for longer tails.
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.fd_window):
            weight *= (lag - 1 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["close_change"] = close.diff()
        out["absolute_change"] = out["close_change"].abs()
        out["change_scale"] = out["absolute_change"].rolling(params.scale_window).mean()
        out["fractional_change"] = fractional.diff()
        out["fractional_impulse"] = out["fractional_change"] / out["change_scale"].replace(0.0, np.nan)
        out["body"] = close - data["open"]
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["body_sum"] = out["body"].rolling(params.body_window).sum()
        out["range_sum"] = out["range"].rolling(params.body_window).sum()
        out["body_efficiency"] = out["body_sum"] / out["range_sum"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        impulse = indicators["fractional_impulse"]
        efficiency = indicators["body_efficiency"]
        long_entry = (impulse > params.impulse_threshold) & (efficiency > params.body_threshold)
        short_entry = (impulse < -params.impulse_threshold) & (efficiency < -params.body_threshold)
        direction = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for bar, qualification in enumerate(direction):
            if position == 0:
                position = int(qualification)
            elif qualification == -position:
                position = 0
            signals[bar] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo():
    params = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=80)
    close = 100.0 + np.arange(80) * 0.1 + np.sin(np.arange(80))
    data = pd.DataFrame({
        "open": close - 0.2, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 1000.0,
    }, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:50], params)
    pd.testing.assert_frame_equal(full.iloc[:50], prefix)
    assert full["fractional_impulse"].iloc[:params.fd_window].isna().all()
    assert strategy.warmup_bars(params) >= params.fd_window + 1
    decisions = pd.DataFrame({
        "fractional_impulse": [1, 1, 0, -1, -1, 0, 1],
        "body_efficiency": [1, -1, 0, -1, -1, 0, 1],
    }, index=index[:7])
    result = strategy.generate_signals(data.iloc[:7], decisions, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert result["signal"].eq(0).all()


if __name__ == "__main__":
    demo()
