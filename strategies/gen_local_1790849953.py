from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    window: int = 8
    topology_gap: float = 0.25
    excursion: float = 0.015
    recovery_fraction: float = 0.35


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local_1790849953"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        if not 5 <= params.window <= 10:
            raise ValueError("window must be between 5 and 10")
        if not 0 < params.topology_gap <= 1:
            raise ValueError("topology_gap must be in (0, 1]")
        if not 0 < params.excursion < 1:
            raise ValueError("excursion must be in (0, 1)")
        if not 0 < params.recovery_fraction < 1:
            raise ValueError("recovery_fraction must be in (0, 1)")
        return int(params.window)

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        n = self.warmup_bars(params)
        prices = np.column_stack([
            data["close"].shift(lag).to_numpy(dtype=float)
            for lag in range(n - 1, -1, -1)
        ])
        valid = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
        prices = np.where(valid[:, None], prices, 1.0)
        transitivity = []
        for direction in (1.0, -1.0):
            y = direction * prices
            graph = np.zeros((len(data), n, n), dtype=bool)
            for a in range(n - 1):
                for b in range(a + 1, n):
                    visible = np.ones(len(data), dtype=bool)
                    for k in range(a + 1, b):
                        line = y[:, a] + (y[:, b] - y[:, a]) * ((k - a) / (b - a))
                        visible &= y[:, k] < line
                    graph[:, a, b] = visible
                    graph[:, b, a] = visible
            triangles = np.zeros(len(data), dtype=float)
            for a in range(n - 2):
                for b in range(a + 1, n - 1):
                    triangles += (
                        graph[:, a, b, None]
                        & graph[:, a, b + 1:]
                        & graph[:, b, b + 1:]
                    ).sum(axis=1)
            degree = graph.sum(axis=2)
            wedges = (degree * (degree - 1) / 2.0).sum(axis=1)
            transitivity.append(np.divide(
                3.0 * triangles, wedges,
                out=np.zeros(len(data), dtype=float), where=wedges > 0
            ))
        low = prices.min(axis=1)
        high = prices.max(axis=1)
        span = high - low
        recovery = np.divide(
            prices[:, -1] - low, span,
            out=np.zeros(len(data), dtype=float), where=span > 0
        )
        result = pd.DataFrame({
            "price_transitivity": transitivity[0],
            "mirror_transitivity": transitivity[1],
            "topology_bias": transitivity[0] - transitivity[1],
            "trough_age": n - 1 - prices.argmin(axis=1),
            "peak_age": n - 1 - prices.argmax(axis=1),
            "excursion": high / low - 1.0,
            "recovery": recovery,
            "retreat": 1.0 - recovery,
        }, index=data.index)
        result.loc[~valid, :] = np.nan
        return result

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: TopologyParams,
    ) -> SignalFrame:
        i = indicators
        long_setup = (
            (i["topology_bias"] >= params.topology_gap)
            & i["trough_age"].between(2, params.window - 2)
            & (i["excursion"] >= params.excursion)
            & (i["recovery"] >= params.recovery_fraction)
        )
        short_setup = (
            (i["topology_bias"] <= -params.topology_gap)
            & i["peak_age"].between(2, params.window - 2)
            & (i["excursion"] >= params.excursion)
            & (i["retreat"] >= params.recovery_fraction)
        )
        desired = np.select([long_setup, short_setup], [1, -1], default=0)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        for bar, entry in enumerate(desired):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            signal[bar] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = TopologyParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 120.0, "low": 90.0,
        "close": [116.0, 109.0, 104.0, 101.0, 100.0, 101.0, 104.0, 109.0],
        "volume": 1000.0,
    }, index=index)
    features = strategy.indicators(data, params)
    assert features.iloc[:-1].isna().all().all()
    assert features.iloc[-1]["topology_bias"] > params.topology_gap
    manual = pd.DataFrame({
        "topology_bias": [1, 1, 0, -1, -1, 0, 1, 1],
        "trough_age": 3, "peak_age": 3, "excursion": 0.1,
        "recovery": 0.5, "retreat": 0.5,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, manual, ctx, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    assert (signals["size"] > 0).all()
