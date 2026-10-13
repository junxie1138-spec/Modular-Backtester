from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilitySeasonParams:
    window: int = 6
    triangle_threshold: float = 0.30
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilitySeasonParams]):
    strategy_id = "gen_local-w2_1791246877"

    @classmethod
    def params_type(cls) -> type[VisibilitySeasonParams]:
        return VisibilitySeasonParams

    def warmup_bars(self, params: VisibilitySeasonParams) -> int:
        if not 3 <= params.window <= 10:
            raise ValueError("window must be between 3 and 10")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not 0.0 < params.triangle_threshold <= 1.0:
            raise ValueError("triangle_threshold must be in (0, 1]")
        return int(params.window)

    def indicators(self, data: pd.DataFrame, params: VisibilitySeasonParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(self, params)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        w = params.window
        nodes = [out["log_close"].shift(w - 1 - i) for i in range(w)]
        upper, lower = {}, {}
        for i in range(w - 1):
            for j in range(i + 1, w):
                up = nodes[i].notna() & nodes[j].notna()
                down = up.copy()
                for k in range(i + 1, j):
                    chord = nodes[i] + (nodes[j] - nodes[i]) * ((k - i) / (j - i))
                    up &= nodes[k] < chord
                    down &= nodes[k] > chord
                upper[i, j] = up
                lower[i, j] = down
        out["upper_triangles"] = 0.0
        out["lower_triangles"] = 0.0
        for i in range(w - 2):
            for j in range(i + 1, w - 1):
                for k in range(j + 1, w):
                    out["upper_triangles"] += (
                        upper[i, j] & upper[j, k] & upper[i, k]
                    ).astype(float)
                    out["lower_triangles"] += (
                        lower[i, j] & lower[j, k] & lower[i, k]
                    ).astype(float)
        total = w * (w - 1) * (w - 2) / 6.0
        out["triangle_balance"] = (
            out["upper_triangles"] - out["lower_triangles"]
        ) / total
        day = data.index.day
        out["calendar_side"] = np.where(
            (day >= 25) | (day <= 3), 1, np.where((day >= 10) & (day <= 16), -1, 0)
        )
        out["valid"] = out["log_close"].rolling(w, min_periods=w).count().eq(w)
        out.loc[~out["valid"], ["upper_triangles", "lower_triangles", "triangle_balance"]] = 0.0
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilitySeasonParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(self, params)
        balance = indicators["triangle_balance"].fillna(0.0)
        side = indicators["calendar_side"].fillna(0)
        valid = indicators["valid"].fillna(False).astype(bool)
        candidates = np.where(
            valid & side.eq(1) & balance.ge(params.triangle_threshold),
            1,
            np.where(valid & side.eq(-1) & balance.le(-params.triangle_threshold), -1, 0),
        )
        signals = np.zeros(len(data), dtype=int)
        position, remaining = 0, 0
        for i, candidate in enumerate(candidates):
            if remaining > 0:
                signals[i] = position
                remaining -= 1
            elif position != 0:
                position = 0
            elif candidate != 0:
                position = int(candidate)
                remaining = params.hold_bars - 1
                signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = VisibilitySeasonParams()
    index = pd.bdate_range("2020-01-01", periods=50)
    x = np.arange(50, dtype=float)
    close = 100.0 * np.exp(0.001 * x + 0.005 * np.sin(x))
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000000.0}, index=index
    )
    features = GeneratedStrategy.indicators(None, data, params)
    prefix = GeneratedStrategy.indicators(None, data.iloc[:30], params)
    pd.testing.assert_frame_equal(features.iloc[:30], prefix)
    assert not features["valid"].iloc[:params.window - 1].any()
    assert features["triangle_balance"].between(-1, 1).all()
    forced = pd.DataFrame(
        {"triangle_balance": 1.0, "calendar_side": 1, "valid": True}, index=index
    )
    for hold in (3, 4, 5):
        params.hold_bars = hold
        for direction in (-1, 1):
            forced["triangle_balance"] = float(direction)
            forced["calendar_side"] = direction
            result = GeneratedStrategy.generate_signals(None, data, forced, None, params).data
            expected = [0] + [
                direction if i % (hold + 1) < hold else 0 for i in range(len(data) - 1)
            ]
            assert result["signal"].tolist() == expected
            assert result["size"].gt(0).all()
    params.hold_bars = 4
    full = GeneratedStrategy.generate_signals(None, data, features, None, params).data
    truncated = GeneratedStrategy.generate_signals(None, data.iloc[:30], prefix, None, params).data
    pd.testing.assert_frame_equal(full.iloc[:30], truncated)


if __name__ == "__main__":
    _self_check()
