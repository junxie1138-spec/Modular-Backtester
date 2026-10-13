from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    lookback: int = 16
    closure_threshold: float = 0.55


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1791243711"

    @classmethod
    def params_type(cls) -> type[TopologyParams]:
        return TopologyParams

    @staticmethod
    def warmup_bars(params: TopologyParams) -> int:
        return max(3, int(params.lookback)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        window = max(3, int(params.lookback))
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        log_close = log_close.replace([np.inf, -np.inf], np.nan)
        out["log_close"] = log_close
        out["return"] = log_close.diff()
        out["impulse_scale"] = out["return"].abs().rolling(window, min_periods=window).mean()

        degree = pd.Series(0.0, index=data.index)
        triangles = pd.Series(0.0, index=data.index)
        maximum_slope = pd.Series(-np.inf, index=data.index)
        previous_visible = pd.Series(False, index=data.index)

        # A past node is naturally visible when its slope exceeds every
        # intervening slope measured from the newest node.
        for lag in range(1, window):
            slope = (log_close.shift(lag) - log_close) / lag
            visible = slope.notna() & slope.gt(maximum_slope)
            degree += visible.astype(float)
            if lag > 1:
                # Shared neighbours close triangles around edge (t-1, t).
                shared = visible & previous_visible.shift(1, fill_value=False)
                triangles += shared.astype(float)
            maximum_slope = pd.Series(
                np.maximum(maximum_slope.to_numpy(), slope.fillna(-np.inf).to_numpy()),
                index=data.index,
            )
            previous_visible = visible

        ready = log_close.rolling(window + 1, min_periods=window + 1).count().eq(window + 1)
        out["degree"] = degree.where(ready)
        out["triangles"] = triangles.where(ready)
        out["closure"] = (triangles / (degree - 1).clip(lower=1)).where(ready)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        threshold = float(np.clip(params.closure_threshold, 0.0, 1.0))
        closure = indicators["closure"]
        impulse = indicators["return"]
        scale = indicators["impulse_scale"]
        valid = closure.notna() & impulse.notna() & scale.notna() & scale.gt(0)
        elastic_regime = closure.ge(threshold)
        recovery = elastic_regime & impulse.lt(-scale)
        continuation = ~elastic_regime & impulse.gt(scale)
        entries = (valid & (recovery | continuation)).fillna(False).to_numpy(dtype=bool)

        signal = np.zeros(len(data), dtype=np.int64)
        entry_bar = -1
        for bar in range(len(data)):
            if entry_bar >= 0:
                if bar - entry_bar == 2:
                    entry_bar = -1
                else:
                    signal[bar] = 1
            elif entries[bar]:
                entry_bar = bar
                signal[bar] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    close = np.array([100.0, 101.0, 99.0, 102.0, 100.0, 103.0, 101.0, 104.0, 102.0])
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)
    params = TopologyParams(lookback=3)
    derived = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:7], params)
    pd.testing.assert_frame_equal(derived.iloc[:7], prefix)
    synthetic = pd.DataFrame({"closure": 0.0, "return": 0.02, "impulse_scale": 0.01}, index=index)
    result = GeneratedStrategy.generate_signals(data, synthetic, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0, 1, 1]
    synthetic["return"] = 0.0
    flat = GeneratedStrategy.generate_signals(data, synthetic, None, params).data
    assert flat["signal"].eq(0).all()
