from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapSlopeParams:
    slope_window: int = 20
    noise_window: int = 60
    innovation_threshold: float = 1.0
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapSlopeParams]):
    strategy_id = "gen_local_1791345287"

    @classmethod
    def params_type(cls):
        return GapSlopeParams

    def warmup_bars(self, params: GapSlopeParams) -> int:
        return max(params.slope_window, params.noise_window) + 2

    def indicators(self, data: pd.DataFrame, params: GapSlopeParams) -> pd.DataFrame:
        if min(params.slope_window, params.noise_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if params.innovation_threshold <= 0 or params.profit_target <= 0 or params.hold_bars < 1:
            raise ValueError("Threshold, target, and holding period must be positive.")
        out = pd.DataFrame(index=data.index)
        out["gap"] = np.log(data["open"].where(data["open"] > 0) / data["close"].shift(1).where(data["close"].shift(1) > 0))
        out["session"] = np.log(data["close"].where(data["close"] > 0) / data["open"].where(data["open"] > 0))
        out = out.replace([np.inf, -np.inf], np.nan)
        n = params.slope_window
        k = np.arange(1, n + 1, dtype=float)
        # OLS slope of cumulative log gaps over n + 1 observations.
        weights = 6.0 * k * (n + 1.0 - k) / (n * (n + 1.0) * (n + 2.0))
        if len(data):
            fitted = np.convolve(out["gap"].fillna(0).to_numpy(), weights, mode="full")[:len(data)]
            out["gap_slope"] = pd.Series(fitted, index=data.index).where(out["gap"].rolling(n).count() == n)
        else:
            out["gap_slope"] = pd.Series(index=data.index, dtype=float)
        out["predicted_slope"] = out["gap_slope"].shift(1)
        out["gap_noise"] = out["gap"].rolling(params.noise_window).std(ddof=0).shift(1)
        out["innovation"] = (out["gap"] - out["predicted_slope"]) / out["gap_noise"].where(out["gap_noise"] > 1e-12)
        out["session_drift"] = out["session"].rolling(n).mean()
        out["conviction"] = ((out["innovation"] - params.innovation_threshold) / 2.0).clip(0.0, 1.0).fillna(0.0)
        out["entry"] = (
            (out["predicted_slope"] < 0)
            & (out["gap"] > 0)
            & (out["innovation"] > params.innovation_threshold)
            & (out["session_drift"] > 0)
            & (out["session"] >= 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapSlopeParams) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        signal = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        conviction = indicators["conviction"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        position_size = 0.25
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Read the fill reference only when its bar arrives.
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        entry_bar = -1
                        continue
                # Close-observed target exits fill on the next bar.
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                timed_out = i - entry_bar + 1 >= params.hold_bars
                if target_hit or timed_out:
                    entry_bar = -1
                    continue
                signal[i] = 1
                sizes[i] = position_size
            elif i >= self.warmup_bars(params) and entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                entry_price = np.nan
                position_size = 0.25 + 0.70 * conviction[i]
                signal[i] = 1
                sizes[i] = position_size
        df["signal"] = signal
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapSlopeParams()
    start = strategy.warmup_bars(params)
    index = pd.date_range("2020-01-01", periods=start + params.hold_bars + 5)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    features = pd.DataFrame({"entry": False, "conviction": 0.5}, index=index)
    features.iloc[start, features.columns.get_loc("entry")] = True
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].sum() == params.hold_bars
    assert result["signal"].iloc[start] == 0
    assert result["signal"].iloc[start + 1] == 1
    assert result["signal"].iloc[start + params.hold_bars + 1] == 0
    assert np.isclose(result["size"].iloc[start + 1], 0.60)
    data.iloc[start + 3, data.columns.get_loc("close")] = 105.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].iloc[start + 3] == 1
    assert result["signal"].iloc[start + 4] == 0
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
