from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 12
    atr_multiple: float = 1.5

    def __post_init__(self):
        if self.pattern_window < 3 or not np.isfinite(self.atr_multiple) or self.atr_multiple <= 0:
            raise ValueError("Require pattern_window >= 3 and finite atr_multiple > 0")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791318927"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.pattern_window + 3, 15)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        c = data["close"]
        a, b = c.shift(2), c.shift(1)
        finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        # Equal prices are ordered by their observation time.
        out["ordinal"] = pd.Series(np.select([
            (a <= b) & (b <= c),
            (a <= c) & (c < b),
            (b < a) & (a <= c),
            (b <= c) & (c < a),
            (c < a) & (a <= b),
            (c < b) & (b < a),
        ], [0, 1, 2, 3, 4, 5], default=np.nan), index=data.index).where(finite)
        source = out["ordinal"].shift(1)
        out["transition"] = source * 6 + out["ordinal"]
        valid = source.notna() & out["ordinal"].notna()
        edges = pd.Series(0.0, index=data.index)
        sources = pd.Series(0.0, index=data.index)
        for code in range(36):
            count = (out["transition"] == code).astype(float).where(valid).rolling(params.pattern_window).sum()
            edges += (count > 0).astype(float).where(count.notna())
        for code in range(6):
            count = (source == code).astype(float).where(valid).rolling(params.pattern_window).sum()
            sources += (count > 0).astype(float).where(count.notna())
        out["edge_count"] = edges
        out["source_count"] = sources
        # Each overlapping three-price ordinal motif has at most three successors.
        out["branching"] = ((edges - sources) / (2 * sources.replace(0, np.nan))).clip(0, 1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - c.shift(1)).abs(),
            (data["low"] - c.shift(1)).abs(),
        ], axis=1).max(axis=1).where(np.isfinite(data[["high", "low", "close"]]).all(axis=1))
        out["atr"] = out["true_range"].rolling(14).mean()
        out["drift"] = c.diff(3)
        span = (data["high"] - data["low"]).replace(0, np.nan)
        out["close_location"] = (c - data["low"]) / span
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        raw = np.zeros(len(data), dtype=int)
        chaotic = True
        active = False
        entry = np.nan
        stop = np.nan
        peak = np.nan
        armed = False
        held = 0
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        features = indicators[["branching", "atr", "drift", "close_location"]].to_numpy(dtype=float)
        for i in range(len(data)):
            op, high, low, close = prices[i]
            complexity, atr, drift, location = features[i]
            previous_regime = chaotic
            if np.isfinite(complexity):
                if complexity >= 0.45:
                    chaotic = True
                elif complexity <= 0.20:
                    chaotic = False
            collapse = previous_regime and not chaotic
            if active:
                if not np.isfinite(prices[i]).all() or not np.isfinite(atr) or atr <= 0:
                    active = False
                    continue
                # Yesterday's unshifted entry decision fills at today's open.
                if np.isnan(entry):
                    entry = op
                    stop = entry - params.atr_multiple * features[i - 1, 1]
                    peak = entry
                held += 1
                # Test the standing stop before raising it using today's high.
                stopped = low <= stop
                peak = max(peak, high)
                if peak >= entry * 1.004:
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.atr_multiple * atr)
                    stopped = stopped or close <= stop
                # Stops are close-time exit decisions, executed next open.
                if stopped or held >= 2:
                    active = False
                else:
                    raw[i] = 1
            elif (collapse and np.isfinite(features[i]).all()
                  and np.isfinite(prices[i]).all() and op > 0 and atr > 0
                  and drift > 0 and location >= 0.65):
                active = True
                entry = np.nan
                stop = np.nan
                peak = np.nan
                armed = False
                held = 0
                raw[i] = 1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 102.8, 103.0],
        "high": [101.0, 102.0, 103.0, 103.2, 104.0],
        "low": [99.0, 100.0, 101.9, 102.7, 102.5],
        "close": [100.5, 101.8, 102.8, 103.0, 103.5],
        "volume": 1000.0,
    }, index=index)
    features = pd.DataFrame({
        "branching": [0.6, 0.1, 0.1, 0.1, 0.1],
        "atr": 1.0, "drift": 1.0, "close_location": 0.8,
    }, index=index)
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    data.loc[index[2], "close"] = 101.95
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0]
    data.loc[index[2], "close"] = 102.8
    data.loc[index[3], "low"] = 101.9
    features.loc[index[3], "atr"] = 10.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) >= params.pattern_window + 3


if __name__ == "__main__":
    _self_check()
