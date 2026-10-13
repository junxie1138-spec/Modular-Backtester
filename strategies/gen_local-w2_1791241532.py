from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 12
    topology_threshold: float = 0.12
    displacement: float = 0.35


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791241532"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return params.graph_window + 1

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if params.graph_window < 4 or not 0 < params.topology_threshold <= 1 or params.displacement < 0:
            raise ValueError("Invalid visibility parameters")
        n = params.graph_window
        close = data["close"].astype(float)
        log_close = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=data.index)
        out["topology"] = np.nan
        out["displacement"] = (log_close - log_close.rolling(n).mean()) / log_close.rolling(n).std(ddof=0).replace(0, np.nan)
        if len(data) < n:
            return out
        windows = np.lib.stride_tricks.sliding_window_view(log_close.to_numpy(), n)
        valid = np.isfinite(windows).all(axis=1)
        count = len(windows)
        upper = np.zeros((count, n, n), dtype=bool)
        lower = np.zeros_like(upper)
        # ponytail: O(bars * window^3), use sparse triangle counting if windows grow beyond a few dozen bars.
        for i in range(n - 1):
            slopes = (windows[:, i + 1:] - windows[:, i, None]) / np.arange(1, n - i)
            upper[:, i, i + 1] = True
            lower[:, i, i + 1] = True
            if slopes.shape[1] > 1:
                upper[:, i, i + 2:] = slopes[:, 1:] > np.maximum.accumulate(slopes[:, :-1], axis=1)
                lower[:, i, i + 2:] = slopes[:, 1:] < np.minimum.accumulate(slopes[:, :-1], axis=1)
        balance = np.zeros(count, dtype=float)
        possible = 0
        for i in range(n - 2):
            for k in range(max(i + 2, i + n // 2), n):
                for j in range(i + 1, k):
                    balance += (upper[:, i, j] & upper[:, j, k] & upper[:, i, k]).astype(float)
                    balance -= (lower[:, i, j] & lower[:, j, k] & lower[:, i, k]).astype(float)
                    possible += 1
        out.iloc[n - 1:, out.columns.get_loc("topology")] = np.where(valid, balance / possible, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        topology = indicators["topology"]
        displacement = indicators["displacement"]
        long_entry = (topology >= params.topology_threshold) & (displacement <= -params.displacement)
        short_entry = (topology <= -params.topology_threshold) & (displacement >= params.displacement)
        raw = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(raw):
            if position:
                if candidate == -position:
                    position = 0
            elif i > 0 and candidate != 0 and candidate == raw[i - 1]:
                position = int(candidate)
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=40, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    for direction in (1, -1):
        sample = data.copy()
        sample["close"] = np.exp(4.0 + direction * 0.002 * (np.arange(40) - 50) ** 2)
        derived = strategy.indicators(sample, params)
        assert np.allclose(derived["topology"].iloc[params.graph_window - 1:], direction)
        assert strategy.generate_signals(sample, derived, None, params).data["signal"].iloc[-1] == direction
        prefix = strategy.indicators(sample.iloc[:25], params)
        pd.testing.assert_frame_equal(prefix, derived.iloc[:25])
    raw = np.array([0, 1, 1, 0, -1, -1, -1, 0])
    scripted = pd.DataFrame({"topology": raw.astype(float), "displacement": -raw.astype(float)}, index=index[:8])
    result = strategy.generate_signals(data.iloc[:8], scripted, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0).all()
    flat = strategy.indicators(data, params)
    assert strategy.generate_signals(data, flat, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
