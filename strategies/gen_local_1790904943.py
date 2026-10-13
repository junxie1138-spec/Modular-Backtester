from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_bars: int = 7
    min_gap: float = 0.002
    min_closure: float = 0.6
    min_agreement: float = 0.6666666666666666
    profit_target: float = 0.012
    max_hold: int = 4
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790904943"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if not 3 <= params.graph_bars <= 9:
            raise ValueError("graph_bars must be between 3 and 9")
        return int(params.graph_bars + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        window = self.warmup_bars(params) - 1
        if not (params.min_gap > 0 and 0 <= params.min_closure <= 1
                and 0 <= params.min_agreement <= 1):
            raise ValueError("Invalid gap or topology thresholds")
        out = pd.DataFrame(index=data.index)
        out["gap"] = (data["open"] / data["close"].shift(1) - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        out["gap_magnitude"] = out["gap"].abs()
        out["intraday"] = (data["close"] / data["open"] - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        # Node zero is today's gap; older nodes are positive lags.
        magnitude = [out["gap_magnitude"].shift(lag) for lag in range(window)]
        edges = {}
        for a in range(window - 1):
            for b in range(a + 1, window):
                visible = magnitude[a].notna() & magnitude[b].notna()
                ceiling = np.minimum(magnitude[a], magnitude[b])
                for k in range(a + 1, b):
                    visible &= magnitude[k] < ceiling
                edges[a, b] = visible

        degree = pd.Series(0.0, index=data.index)
        agreeing = degree.copy()
        triangles = degree.copy()
        polarity = np.sign(out["gap"])
        for a in range(1, window):
            neighbor = edges[0, a]
            degree += neighbor.astype(float)
            agreeing += (neighbor & (np.sign(out["gap"].shift(a)) == polarity)).astype(float)
            for b in range(a + 1, window):
                triangles += (neighbor & edges[0, b] & edges[a, b]).astype(float)

        pairs = degree * (degree - 1) / 2
        out["degree"] = degree
        out["triangles"] = triangles
        out["closure"] = (triangles / pairs.replace(0, np.nan)).fillna(0.0)
        out["agreement"] = (agreeing / degree.replace(0, np.nan)).fillna(0.0)
        out["ready"] = out["gap"].rolling(window).count().eq(window)
        qualified = (
            out["ready"] & (degree >= 2)
            & (out["gap_magnitude"] >= params.min_gap)
            & (out["closure"] >= params.min_closure)
            & (out["agreement"] >= params.min_agreement)
            & (out["intraday"] * out["gap"] > 0)
        )
        out["entry"] = polarity.where(qualified, 0).fillna(0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        if not (3 <= params.max_hold <= 5 and params.profit_target > 0
                and 0 < params.size <= 1):
            raise ValueError("Invalid holding period, profit target, or size")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                if not (np.isfinite(entry_price) and entry_price > 0
                        and np.isfinite(closes[i]) and closes[i] > 0):
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1)
                    # Close-observed exits execute at the following open.
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.max_hold:
                        position = 0
            elif entries[i] in (-1, 1):
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    features = strategy.indicators(data, params)
    assert strategy.warmup_bars(params) == 8
    assert features["entry"].eq(0).all()
    features["entry"] = 0
    features.iloc[0, features.columns.get_loc("entry")] = 1
    features.iloc[6, features.columns.get_loc("entry")] = -1
    data.iloc[7, data.columns.get_loc("close")] = 98.0
    signal = strategy.generate_signals(data, features, ctx, params).data
    assert signal["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, -1, 0, 0, 0, 0]
    assert signal["size"].gt(0).all()
    # Three mutually visible endpoint nodes form one triangle.
    gaps = [0.005, 0.001, 0.004]
    sample = data.iloc[:4].copy()
    sample["close"] = 100.0
    sample["open"] = [100.0] + [100.0 * (1 + gap) for gap in gaps]
    topology = strategy.indicators(sample, GeneratedParams(graph_bars=3))
    assert topology["triangles"].iloc[-1] == 1
    assert topology["closure"].iloc[-1] == 1


if __name__ == "__main__":
    _self_check()
