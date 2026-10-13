from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_bars: int = 40
    z_threshold: float = 1.5
    drift_bars: int = 5
    max_center_drift: float = 0.5
    hold_bars: int = 8

    def __post_init__(self):
        if self.ma_bars < 2 or self.drift_bars < 1 or self.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive; ma_bars must be >= 2.")
        if not np.isfinite(self.z_threshold) or self.z_threshold <= 0:
            raise ValueError("z_threshold must be finite and positive.")
        if not np.isfinite(self.max_center_drift) or self.max_center_drift < 0:
            raise ValueError("max_center_drift must be finite and nonnegative.")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790966592"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.ma_bars + params.drift_bars + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        window = close.rolling(params.ma_bars, min_periods=params.ma_bars)
        out = pd.DataFrame(index=data.index)
        out["ma"] = window.mean()
        out["scale"] = window.std(ddof=0).replace(0.0, np.nan)
        # Freeze the equilibrium before either confirmation close.
        out["anchor"] = out["ma"].shift(2)
        out["anchor_scale"] = out["scale"].shift(2)
        out["center_drift"] = out["ma"].diff(params.drift_bars).shift(2).abs() / out["anchor_scale"]
        out["prior_z"] = (close.shift(2) - out["anchor"]) / out["anchor_scale"]
        out["first_z"] = (close.shift(1) - out["anchor"]) / out["anchor_scale"]
        out["second_z"] = (close - out["anchor"]) / out["anchor_scale"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        t = params.z_threshold
        ready = (
            indicators[["prior_z", "first_z", "second_z", "center_drift"]].notna().all(axis=1)
            & indicators["prior_z"].abs().le(t)
            & indicators["center_drift"].le(params.max_center_drift)
        )
        long_entry = (
            ready
            & indicators["first_z"].gt(t)
            & indicators["second_z"].gt(indicators["first_z"])
        )
        short_entry = (
            ready
            & indicators["first_z"].lt(-t)
            & indicators["second_z"].lt(indicators["first_z"])
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
                # Do not replace the scheduled flat bar with a new entry.
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.hold_bars
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams(ma_bars=20)
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for direction in (1, -1):
        close = 100.0 + direction * np.array([-1.0, 1.0] * 25 + [3.0, 4.0, -3.0, -4.0] + [0.0] * 12)
        data = pd.DataFrame(
            {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0},
            index=pd.date_range("2020-01-01", periods=len(close), freq="B"),
        )
        indicators = strategy.indicators(data, params)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].iloc[:52].eq(0).all()
        assert result["signal"].iloc[52:60].eq(direction).all()
        assert result["signal"].iloc[60:].eq(0).all()
        assert result["size"].gt(0).all()
        prefix = data.iloc[:56]
        partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
        assert partial.equals(result.iloc[:56])
    flat = data.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    assert strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
