from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    fractional_window: int = 32
    baseline_window: int = 40
    channel_window: int = 30
    excursion: float = 0.025
    recovery_z: float = 0.4
    hold_bars: int = 8

    def __post_init__(self):
        if not 0.0 < self.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if min(self.fractional_window, self.baseline_window, self.channel_window) < 2:
            raise ValueError("lookback windows must be at least two")
        if not 0.0 < self.excursion < 1.0 or not 0.0 < self.recovery_z < 10.0:
            raise ValueError("invalid excursion or recovery_z")
        if not 5 <= self.hold_bars <= 10:
            raise ValueError("hold_bars must be between five and ten")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791200026"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.fractional_window + params.baseline_window + 1,
                       params.channel_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0.0))
        # Fixed-width causal fractional difference: w[k] = -w[k-1]*(d-k+1)/k.
        fd = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fd = fd + weight * close.shift(lag)
        baseline = fd.rolling(params.baseline_window, min_periods=params.baseline_window)
        scale = baseline.std(ddof=0).replace(0.0, np.nan)
        fd_z = ((fd - baseline.mean()) / scale).replace([np.inf, -np.inf], np.nan)
        fd_step = fd.diff()
        peak = close.rolling(params.channel_window, min_periods=params.channel_window).max()
        trough = close.rolling(params.channel_window, min_periods=params.channel_window).min()
        drawdown = 1.0 - close / peak
        drawup = close / trough - 1.0
        out = pd.DataFrame({
            "fractional_close": fd,
            "fractional_z": fd_z,
            "fractional_step": fd_step,
            "drawdown": drawdown,
            "drawup": drawup,
        }, index=data.index)
        out["long_setup"] = (
            drawdown.ge(params.excursion) & fd_z.gt(params.recovery_z) & fd_step.gt(0.0)
        ).fillna(False)
        out["short_setup"] = (
            drawup.ge(params.excursion) & fd_z.lt(-params.recovery_z) & fd_step.lt(0.0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        long_setup = indicators["long_setup"].fillna(False).astype(bool)
        short_setup = indicators["short_setup"].fillna(False).astype(bool)
        long_entry = (long_setup & long_setup.shift(1, fill_value=False)).to_numpy()
        short_entry = (short_setup & short_setup.shift(1, fill_value=False)).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entered_at = -1
        for i in range(len(data)):
            if side:
                # Both entry and exit move one bar later after the mandatory shift.
                if i - entered_at == params.hold_bars:
                    side = 0
            elif long_entry[i] and not short_entry[i]:
                side, entered_at = 1, i
            elif short_entry[i] and not long_entry[i]:
                side, entered_at = -1, i
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    setups = pd.DataFrame(False, index=index, columns=["long_setup", "short_setup"])
    setups.iloc[0:2, 0] = True
    setups.iloc[12:14, 1] = True
    setups.iloc[26, 0] = True
    result = strategy.generate_signals(data, setups, None, params).data
    expected = np.zeros(30, dtype=int)
    expected[2:10] = 1
    expected[14:22] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0.0).all()
    assert strategy.indicators(data, params)["long_setup"].eq(False).all()
