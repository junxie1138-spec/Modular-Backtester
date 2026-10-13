from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeTerritoryParams:
    range_window: int = 20
    territory_window: int = 5
    flow_window: int = 5
    atr_window: int = 14
    expansion_ratio: float = 1.05
    territory_threshold: float = 0.25
    flow_threshold: float = 0.12
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold_bars: int = 10


class GeneratedStrategy(BaseStrategy[RangeTerritoryParams]):
    strategy_id = "gen_local_1790966111"

    @classmethod
    def params_type(cls):
        return RangeTerritoryParams

    def warmup_bars(self, params: RangeTerritoryParams) -> int:
        return max(params.range_window + 1, params.territory_window + 1,
                   params.flow_window, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: RangeTerritoryParams) -> pd.DataFrame:
        if min(params.range_window, params.territory_window, params.flow_window,
               params.atr_window, params.max_hold_bars) < 1:
            raise ValueError("Windows and holding period must be positive.")
        if params.trail_atr <= 0 or params.breakeven_pct <= 0:
            raise ValueError("Stop parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["upper_extension"] = np.minimum(
            data["high"].diff().clip(lower=0), out["range"])
        out["lower_extension"] = np.minimum(
            (-data["low"].diff()).clip(lower=0), out["range"])
        out["upper_total"] = out["upper_extension"].rolling(params.territory_window).sum()
        out["lower_total"] = out["lower_extension"].rolling(params.territory_window).sum()
        total = out["upper_total"] + out["lower_total"]
        out["territory_balance"] = (
            (out["upper_total"] - out["lower_total"]) / total.replace(0, np.nan))
        out["range_baseline"] = out["range"].shift(1).rolling(params.range_window).mean()
        out["range_ratio"] = out["range"] / out["range_baseline"].replace(0, np.nan)
        out["body_fraction"] = (
            (data["close"] - data["open"]) / out["range"].replace(0, np.nan)).fillna(0)
        out["volume"] = data["volume"].clip(lower=0)
        out["body_flow"] = out["body_fraction"] * out["volume"]
        out["flow_total"] = out["body_flow"].rolling(params.flow_window).sum()
        out["volume_total"] = out["volume"].rolling(params.flow_window).sum()
        out["flow_balance"] = out["flow_total"] / out["volume_total"].replace(0, np.nan)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeTerritoryParams) -> SignalFrame:
        range_agrees = (
            (indicators["range_ratio"] >= params.expansion_ratio)
            & (indicators["territory_balance"] >= params.territory_threshold))
        flow_agrees = indicators["flow_balance"] >= params.flow_threshold
        entries = (range_agrees & flow_agrees).fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry = peak = stop = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            # The previous close's entry instruction fills at today's open.
            if not active and i > 0 and signals[i - 1] == 1:
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry = peak = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    active = False
                    continue
                peak = max(peak, closes[i])
                armed = armed or peak >= entry * (1.0 + params.breakeven_pct)
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # Close-observed exits execute at the following open.
                if closes[i] <= stop or age >= params.max_hold_bars:
                    active = False
                    continue
                signals[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeTerritoryParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=7)
    close = np.array([100.0, 100.0, 103.0, 102.0, 100.5, 100.0, 100.0])
    data = pd.DataFrame({"open": 100.0, "high": close + 1.0,
                         "low": np.minimum(close, 100.0) - 1.0,
                         "close": close, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"range_ratio": [2.0] + [0.0] * 6,
                        "territory_balance": 0.8, "flow_balance": 0.5,
                        "atr": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0]}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    params.max_hold_bars = 2
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    ind["flow_balance"] = 0.0
    assert not strategy.generate_signals(data, ind, ctx, params).data["signal"].any()
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert not strategy.generate_signals(data, actual, ctx, params).data["signal"].any()
