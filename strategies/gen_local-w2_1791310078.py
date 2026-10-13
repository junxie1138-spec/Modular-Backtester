from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    match_radius: float = 0.65
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791310078"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return 286

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if not all(np.isfinite(v) and v > 0 for v in (params.match_radius, params.trail_atr)):
            raise ValueError("Parameters must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(20, min_periods=20).mean()
        out["std"] = close.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan)
        out["z"] = ((close - out["ma"]) / out["std"]).replace([np.inf, -np.inf], np.nan)
        out["z_prev"] = out["z"].shift(1)
        previous = close.shift(1)
        out["tr"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["tr"].rolling(14, min_periods=14).mean()
        offsets = (0, 2, 4, 6, 8)
        shape = []
        for offset in offsets:
            out[f"shape_{offset}"] = out["z"].shift(offset) - out["z"]
            shape.append(out[f"shape_{offset}"])
        forward_sum = pd.Series(0.0, index=data.index)
        reverse_sum = forward_sum.copy()
        forward_weight = forward_sum.copy()
        reverse_weight = forward_sum.copy()
        forward_count = forward_sum.copy()
        reverse_count = forward_sum.copy()
        # Eight-session spacing reduces overlapping outcomes. Every label
        # matures at least ten bars before the current decision.
        for lag in range(18, 259, 8):
            endpoint = out["z"].shift(lag)
            oldest = out["z"].shift(lag + 8)
            level_error = (out["z"] - endpoint).pow(2)
            forward_error = level_error.copy()
            reverse_error = level_error.copy()
            for offset, current_shape in zip(offsets, shape):
                historical = out["z"].shift(lag + offset) - endpoint
                reversed_shape = out["z"].shift(lag + 8 - offset) - oldest
                forward_error = forward_error + (current_shape - historical).pow(2) / 5.0
                reverse_error = reverse_error + (current_shape - reversed_shape).pow(2) / 5.0
            outcome = (close.shift(lag - 8) / close.shift(lag) - 1.0).replace(
                [np.inf, -np.inf], np.nan
            )
            fw = (1.0 - np.sqrt(forward_error) / params.match_radius).clip(lower=0.0)
            rw = (1.0 - np.sqrt(reverse_error) / params.match_radius).clip(lower=0.0)
            fw = fw.where(outcome.notna(), 0.0).fillna(0.0)
            rw = rw.where(outcome.notna(), 0.0).fillna(0.0)
            forward_sum += fw * outcome.fillna(0.0)
            reverse_sum += rw * outcome.fillna(0.0)
            forward_weight += fw
            reverse_weight += rw
            forward_count += (fw > 0.0).astype(int)
            reverse_count += (rw > 0.0).astype(int)
        out["forward_count"] = forward_count
        out["reverse_count"] = reverse_count
        out["forward_return"] = forward_sum / forward_weight.replace(0.0, np.nan)
        out["reverse_return"] = reverse_sum / reverse_weight.replace(0.0, np.nan)
        out["order_edge"] = out["forward_return"] - out["reverse_return"]
        out["ready"] = np.arange(len(data)) >= self.warmup_bars(params) - 1
        out["entry"] = (
            out["ready"] & out["z"].between(-1.75, -0.15)
            & (out["z"] > out["z_prev"])
            & (out["forward_count"] >= 4) & (out["reverse_count"] >= 4)
            & (out["forward_return"] > 0.004) & (out["order_edge"] > 0.003)
            & np.isfinite(out["atr"]) & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        holding = False
        entry_bar = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if holding:
                if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                    holding = False
                else:
                    # Only closes on or after the next-bar entry contribute.
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    if close[i] <= stop or i - entry_bar + 1 >= 8:
                        holding = False
            elif entry[i] and np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0.0:
                holding = True
                entry_bar = i + 1
                high_water = -np.inf
                stop = -np.inf
            raw[i] = int(holding)
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"close": [120, 100, 102, 104, 103, 101, 100, 100, 100, 100, 100, 100]}, index=index)
    indicators = pd.DataFrame({"atr": [1, 1, 1, 1, 10, 10, 1, 1, 1, 1, 1, 1],
                               "entry": [True] + [False] * 11}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=286)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    data["close"] = 100.0
    indicators["atr"] = 1.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0] + [1] * 8 + [0] * 3
    assert (result["size"] > 0.0).all()
