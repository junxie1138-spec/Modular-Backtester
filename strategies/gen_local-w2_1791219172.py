from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 8
    atr_window: int = 10
    closure_threshold: float = 0.35
    impulse_threshold: float = 0.20
    size_scale: float = 0.70


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local-w2_1791219172"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(3, params.graph_window, max(1, params.atr_window) + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(max(1, int(params.atr_window))).mean()
        out["impulse"] = close.diff() / out["atr"].where(out["atr"] > 0)
        out["closure"] = np.nan
        width = max(3, int(params.graph_window))
        if len(data) < width:
            return out

        prices = close.where(close > 0).to_numpy()
        windows = np.lib.stride_tricks.sliding_window_view(prices, width)
        valid = np.isfinite(windows).all(axis=1)
        count = len(windows)
        edges = np.zeros((count, width, width), dtype=bool)
        for i in range(width - 1):
            for j in range(i + 1, width):
                if j == i + 1:
                    visible = np.ones(count, dtype=bool)
                else:
                    barrier = windows[:, i + 1:j].max(axis=1)
                    visible = barrier < np.minimum(windows[:, i], windows[:, j])
                edges[:, i, j] = visible
                edges[:, j, i] = visible

        degree = edges.sum(axis=2).astype(float)
        wedges = (degree * (degree - 1.0) / 2.0).sum(axis=1)
        triangles = np.zeros(count, dtype=float)
        for i in range(width - 2):
            for j in range(i + 1, width - 1):
                for k in range(j + 1, width):
                    triangles += edges[:, i, j] & edges[:, j, k] & edges[:, i, k]
        closure = np.divide(3.0 * triangles, wedges, out=np.zeros(count), where=wedges > 0)
        closure[~valid] = np.nan
        out.iloc[width - 1:, out.columns.get_loc("closure")] = closure
        out["reverting_regime"] = out["closure"].ge(params.closure_threshold)
        out["forecast"] = out["impulse"].where(~out["reverting_regime"], -out["impulse"])
        out["topology_confidence"] = (out["closure"] - params.closure_threshold).abs().clip(0.0, 1.0)
        out["conviction"] = (out["forecast"].abs() * (1.0 + out["topology_confidence"]) * max(0.0, params.size_scale)).clip(0.15, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityParams) -> SignalFrame:
        forecast = indicators.get("forecast", pd.Series(np.nan, index=data.index))
        closure = indicators["closure"]
        conviction = indicators.get("conviction", pd.Series(np.nan, index=data.index))
        valid = forecast.notna() & closure.notna() & conviction.notna()
        valid.iloc[:self.warmup_bars(params)] = False
        active = valid & forecast.abs().ge(max(0.0, params.impulse_threshold))
        desired = pd.Series(0, index=data.index, dtype=int)
        desired.loc[active & forecast.gt(0)] = 1
        desired.loc[active & forecast.lt(0)] = -1
        signals, sizes = self._positions(desired.to_numpy(), conviction.fillna(0.15).to_numpy())
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.15).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

    @staticmethod
    def _positions(desired: np.ndarray, conviction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signals = np.zeros(len(desired), dtype=int)
        sizes = np.full(len(desired), 0.15, dtype=float)
        position = 0
        entry_size = 0.15
        for i, direction in enumerate(desired):
            if position != 0:
                if direction == -position:
                    position = 0
                    entry_size = 0.15
            elif direction != 0:
                position = int(direction)
                entry_size = float(np.clip(conviction[i], 0.15, 0.95))
            signals[i] = position
            sizes[i] = entry_size
        return signals, sizes


if __name__ == "__main__":
    signals, sizes = GeneratedStrategy._positions(
        np.array([1, 0, -1, -1, 0, 1, 1]),
        np.array([0.4, 0.8, 0.7, 0.6, 0.3, 0.9, 0.5]),
    )
    assert signals.tolist() == [1, 1, 0, -1, -1, 0, 1]
    assert np.allclose(sizes, [0.4, 0.4, 0.15, 0.6, 0.6, 0.15, 0.5])
