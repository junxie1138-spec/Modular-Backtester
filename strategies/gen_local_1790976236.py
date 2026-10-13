from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    graph_window: int = 12
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790976236"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.graph_window + 1, 21)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.graph_window < 4 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("graph_window must be >= 4 and trail_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        ranges = (data["high"] - data["low"]) / close.where(close > 0)
        previous = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = tr.where(previous.notna()).rolling(14, min_periods=14).mean()
        out["relative_range"] = ranges
        out["compression"] = (
            ranges.rolling(3, min_periods=3).mean()
            / ranges.rolling(20, min_periods=20).mean().replace(0, np.nan)
        ).shift(1)
        n = params.graph_window
        values = [ranges.shift(n - 1 - i) for i in range(n)]
        count = pd.Series(0.0, index=data.index)
        pairs = 0
        # Natural visibility: intermediate ranges must lie strictly below the chord.
        # ponytail: O(n^3) shifted comparisons; use compiled graph code if windows grow large.
        for left in range(n // 2):
            for right in range(n // 2, n):
                if right - left < 2:
                    continue
                visible = values[left].notna() & values[right].notna()
                for middle in range(left + 1, right):
                    chord = values[left] + (
                        (values[right] - values[left]) * (middle - left) / (right - left)
                    )
                    visible &= values[middle].notna() & (values[middle] < chord)
                count += visible.astype(float)
                pairs += 1
        valid = ranges.rolling(n, min_periods=n).count().eq(n)
        out["bridge_density"] = (count / pairs).where(valid)
        out["prior_high"] = data["high"].rolling(3, min_periods=3).max().shift(1)
        out["entry"] = (
            (out["bridge_density"].shift(1) >= 0.65)
            & (out["compression"] < 0.8)
            & (close > out["prior_high"])
            & (close > data["open"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        peak = np.nan
        stop = np.nan
        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if active:
                if not valid:
                    active = False
                else:
                    # First observed in-trade close is on the next-bar fill day.
                    peak = close[i] if not np.isfinite(peak) else max(peak, close[i])
                    candidate = peak - params.trail_atr * atr[i]
                    stop = candidate if not np.isfinite(stop) else max(stop, candidate)
                    if close[i] <= stop or i - entry_bar >= 10:
                        active = False
            elif valid and entry[i]:
                active = True
                entry_bar = i
                peak = np.nan
                stop = np.nan
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=40)
    close = 100 + np.sin(np.arange(40) / 3)
    data = pd.DataFrame({
        "open": close - 0.1, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 1000.0,
    }, index=index)
    features = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(
        features.iloc[:30], strategy.indicators(data.iloc[:30], params)
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    test_data = data.iloc[:16].copy()
    test_data["close"] = [100, 101, 104, 103, 101] + [101] * 11
    test_features = pd.DataFrame({
        "atr": [1, 1, 1, 10, 10] + [10] * 11,
        "entry": [True] + [False] * 15,
    }, index=test_data.index)
    result = strategy.generate_signals(test_data, test_features, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1] + [0] * 11
    test_data["close"] = 100.0
    test_features["atr"] = 1.0
    result = strategy.generate_signals(test_data, test_features, ctx, params).data
    assert result["signal"].sum() == 10 and result["signal"].iloc[11] == 0
    assert result["size"].gt(0).all() and result["signal"].isin([0, 1]).all()
