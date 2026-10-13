from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    capacity: float = 1.5
    allowance: float = 0.25
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791257257"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not np.isfinite(params.capacity) or params.capacity <= 0:
            raise ValueError("capacity must be finite and positive")
        if not np.isfinite(params.allowance) or params.allowance < 0:
            raise ValueError("allowance must be finite and nonnegative")
        if isinstance(params.hold_bars, bool) or not isinstance(params.hold_bars, int) or not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be an integer between 5 and 10")
        return 10

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_price"] = np.log(close.where(np.isfinite(close) & (close > 0)))
        y = out["log_price"]
        out["log_return"] = y.diff()
        # Exact five-point OLS slope: centered time weights are -2,-1,0,1,2.
        out["slope"] = (2 * y + y.shift(1) - y.shift(3) - 2 * y.shift(4)) / 10.0
        out["slope_change"] = out["slope"].diff()
        out["prior_noise"] = out["log_return"].rolling(5, min_periods=5).std(ddof=0).shift(1)
        noise = out["prior_noise"].where(out["prior_noise"] > 1e-12)
        out["innovation"] = out["slope_change"] * np.sqrt(10.0) / noise
        out["excess_innovation"] = (out["innovation"] - params.allowance).clip(lower=0.0)
        out["load"] = out["excess_innovation"].rolling(3, min_periods=3).sum()
        out["prior_load"] = out["load"].shift(1)
        out["prior_slope_min"] = out["slope"].shift(1).rolling(3, min_periods=3).min()
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["load"] > params.capacity)
            & (out["prior_load"] <= params.capacity)
            & (out["slope"] > 0)
            & (out["prior_slope_min"] <= 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].reindex(data.index).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    # Force a flat bar at expiry; entries cannot extend the hold.
                    exit_bar = -1
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    close = np.exp(4.0 + np.cumsum(np.sin(np.arange(40)) * 0.01))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    derived = strategy.indicators(data, params)
    assert derived["prior_load"].first_valid_index() == index[9]
    pd.testing.assert_frame_equal(derived.iloc[:25], strategy.indicators(data.iloc[:25], params))
    entries = pd.DataFrame({"entry": True}, index=index)
    result = strategy.generate_signals(data, entries, None, params).data
    assert result["signal"].iloc[:19].tolist() == [0] + [1] * 8 + [0] + [1] * 8 + [0]
    assert set(result["signal"].unique()) <= {0, 1}
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) == 10


if __name__ == "__main__":
    _self_check()
