from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    lookback: int = 40
    min_streak: int = 3
    impact_decay: float = 0.6
    extension_min: float = 1.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791269511"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(params.lookback + 2, params.min_streak + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.lookback < 2 or params.min_streak < 2 or params.hold_bars < 1:
            raise ValueError("Lookback and streak must be >= 2; holding bars must be positive.")
        if not 0.0 < params.impact_decay < 1.0 or params.extension_min <= 0.0:
            raise ValueError("Impact decay must be in (0, 1); extension minimum must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        volume = data["volume"].where(data["volume"] > 0)
        out["return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["return"]).fillna(0).astype(int)
        out["run_id"] = out["direction"].ne(out["direction"].shift(1)).cumsum()
        groups = out["run_id"]
        out["streak"] = (out.groupby("run_id").cumcount() + 1).where(out["direction"].ne(0), 0)
        out["volume_base"] = volume.rolling(params.lookback, min_periods=params.lookback).mean().shift(1)
        out["relative_volume"] = volume / out["volume_base"].where(out["volume_base"] > 0)
        out["volatility"] = out["return"].rolling(params.lookback, min_periods=params.lookback).std(ddof=0).shift(1)
        out["impact"] = out["return"].abs() / out["relative_volume"].where(out["relative_volume"] > 0)
        out["initial_impact"] = out["impact"].where(out["streak"].eq(1)).groupby(groups).ffill()
        out["impact_ratio"] = out["impact"] / out["initial_impact"].where(out["initial_impact"] > 0)
        out["run_return"] = out["return"].fillna(0).groupby(groups).cumsum()
        out["extension"] = out["run_return"].abs() / (
            out["volatility"].where(out["volatility"] > 0) * np.sqrt(out["streak"].where(out["streak"] > 0))
        )
        out["eligible"] = (
            out["streak"].ge(params.min_streak)
            & out["relative_volume"].ge(1.0)
            & out["impact_ratio"].le(params.impact_decay)
            & out["extension"].ge(params.extension_min)
        )
        out["setup"] = (-out["direction"]).where(out["eligible"], 0).astype(int)
        out["conviction"] = (
            (1.0 - out["impact_ratio"].clip(0.0, 1.0))
            * (out["streak"] / params.min_streak).clip(upper=2.0)
            * (out["extension"] / params.extension_min).clip(upper=2.0)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["entry_size"] = (0.25 + 0.75 * out["conviction"].clip(0.0, 1.0)).astype(float)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StreakParams,
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("Holding bars must be positive.")
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        setups = indicators["setup"].fillna(0).to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].fillna(0.25).clip(0.25, 1.0).to_numpy(dtype=float)
        position = 0
        exit_bar = -1
        position_size = 0.25
        for i in range(self.warmup_bars(params), len(data)):
            if position != 0 and i == exit_bar:
                position = 0
                continue
            if position == 0 and setups[i] != 0:
                position = int(setups[i])
                position_size = float(entry_sizes[i])
                exit_bar = i + params.hold_bars
            signals[i] = position
            if position != 0:
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        # Shift row values without changing timestamps: https://pandas.pydata.org/docs/reference/api/pandas.Series.shift.html
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = StreakParams()
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=84, freq="B")
    returns = np.resize(np.array([0.005, -0.005]), 84)
    returns[44:47] = [0.012, 0.010, 0.003]
    close = 100.0 * np.exp(np.cumsum(returns))
    volume = np.full(84, 100.0)
    volume[44:47] = [110.0, 140.0, 200.0]
    data = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": volume,
    }, index=index)
    actual = strategy.indicators(data, params)
    assert actual["streak"].iloc[46] == 3
    assert actual["setup"].iloc[46] == -1
    pd.testing.assert_frame_equal(actual.iloc[:60], strategy.indicators(data.iloc[:60], params))
    synthetic = pd.DataFrame({"setup": 0, "entry_size": 0.25}, index=index)
    synthetic.iloc[43] = [1, 0.65]
    synthetic.iloc[44] = [-1, 0.95]
    synthetic.iloc[61] = [-1, 0.95]
    synthetic.iloc[62] = [-1, 0.85]
    result = strategy.generate_signals(data, synthetic, None, params).data
    assert result["signal"].iloc[:44].eq(0).all()
    assert result["signal"].iloc[44:62].eq(1).all()
    assert result["size"].iloc[44:62].eq(0.65).all()
    assert result["signal"].iloc[62] == 0
    assert result["signal"].iloc[63:81].eq(-1).all()
    assert result["size"].iloc[63:81].eq(0.85).all()
    assert result["signal"].iloc[81] == 0
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
