from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    topology_distance: int = 0
    min_analog_return: float = 0.003
    hold_bars: int = 7


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790868296"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        if not 0 <= params.topology_distance <= 3:
            raise ValueError("topology_distance must be between 0 and 3")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between 5 and 10")
        if not np.isfinite(params.min_analog_return) or params.min_analog_return < 0:
            raise ValueError("min_analog_return must be finite and nonnegative")
        return 10

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        price = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        price = price.where(price > 0)
        out["clean_close"] = price
        points = [price.shift(3 - j) for j in range(4)]
        valid = pd.concat(points, axis=1).notna().all(axis=1)
        out["valid_shape"] = valid
        # Natural visibility: every intermediate close lies strictly below the chord.
        edges = []
        for left, right in ((0, 2), (1, 3), (0, 3)):
            visible = valid.copy()
            for middle in range(left + 1, right):
                chord = points[left] + (points[right] - points[left]) * (
                    (middle - left) / (right - left)
                )
                visible &= points[middle] < chord
            name = f"edge_{left}_{right}"
            out[name] = visible.astype(float).where(valid)
            edges.append(name)
        out["motif"] = out[edges[0]] + 2 * out[edges[1]] + 4 * out[edges[2]]
        for lag in (4, 6):
            comparable = valid & valid.shift(lag, fill_value=False)
            distance = sum(
                (out[name] != out[name].shift(lag)).astype(float)
                for name in edges
            )
            out[f"distance_{lag}"] = distance.where(comparable)
            # Each analog's three-bar outcome is completely observed before today.
            out[f"analog_return_{lag}"] = price.shift(lag - 3) / price.shift(lag) - 1
        out["forecast"] = (
            out["analog_return_4"] + out["analog_return_6"]
        ) / 2
        out["shape_return"] = price / price.shift(3) - 1
        out["last_return"] = price / price.shift(1) - 1
        out["entry"] = (
            (out["distance_4"] <= params.topology_distance)
            & (out["distance_6"] <= params.topology_distance)
            & (out["analog_return_4"] > 0)
            & (out["analog_return_6"] > 0)
            & (out["forecast"] > params.min_analog_return)
            & (out["shape_return"] < 0)
            & (out["last_return"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, entry in enumerate(entries):
            if i < exit_at:
                signal[i] = 1
            elif i == exit_at:
                continue
            elif entry:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=32, freq="B")
    close = np.full(len(index), 100.0)
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    derived = strategy.indicators(data, params)
    assert not derived["entry"].any()
    assert derived["distance_6"].iloc[:9].isna().all()
    forced = pd.DataFrame({"entry": True}, index=index)
    result = strategy.generate_signals(data, forced, ctx, params).data
    expected = [0] + [
        int(i % (params.hold_bars + 1) < params.hold_bars)
        for i in range(len(data) - 1)
    ]
    assert result["signal"].tolist() == expected
    assert result["size"].gt(0).all()
    prefix = strategy.indicators(data.iloc[:20], params)
    pd.testing.assert_frame_equal(derived.iloc[:20], prefix)


if __name__ == "__main__":
    _self_check()
