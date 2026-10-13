from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityValleyParams:
    graph_window: int = 15
    flow_window: int = 7
    topology_threshold: float = 0.15
    flow_threshold: float = 0.05


class GeneratedStrategy(BaseStrategy[VisibilityValleyParams]):
    strategy_id = "gen_local_1791240728"

    @classmethod
    def params_type(cls) -> type[VisibilityValleyParams]:
        return VisibilityValleyParams

    def warmup_bars(self, params: VisibilityValleyParams) -> int:
        return int(max(params.graph_window + 3, params.flow_window))

    def indicators(self, data: pd.DataFrame, params: VisibilityValleyParams) -> pd.DataFrame:
        if params.graph_window < 2 or params.flow_window < 1:
            raise ValueError("Lookbacks must be positive, with graph_window >= 2")
        if not (0.0 <= params.topology_threshold <= 1.0):
            raise ValueError("topology_threshold must be between zero and one")
        if not (0.0 <= params.flow_threshold <= 1.0):
            raise ValueError("flow_threshold must be between zero and one")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where((close > 0) & np.isfinite(close)))
        x = out["log_close"]
        upper_degree = np.zeros(len(data), dtype=float)
        lower_degree = np.zeros(len(data), dtype=float)
        upper_record = np.full(len(data), -np.inf)
        lower_record = np.full(len(data), -np.inf)

        # A past vertex is visible iff its endpoint slope exceeds every nearer slope.
        # ponytail: O(N * graph_window), use a specialized graph algorithm if windows become large.
        for lag in range(1, params.graph_window + 1):
            slope = ((x.shift(lag) - x) / lag).to_numpy()
            upper_degree += (slope > upper_record).astype(float)
            lower_degree += (-slope > lower_record).astype(float)
            upper_record = np.maximum(upper_record, slope)
            lower_record = np.maximum(lower_record, -slope)

        graph_ready = x.rolling(params.graph_window + 1, min_periods=params.graph_window + 1).count().eq(params.graph_window + 1)
        out["upper_degree"] = pd.Series(upper_degree, index=data.index).where(graph_ready)
        out["lower_degree"] = pd.Series(lower_degree, index=data.index).where(graph_ready)
        out["topology_raw"] = (out["lower_degree"] - out["upper_degree"]) / params.graph_window
        out["topology"] = out["topology_raw"].rolling(3, min_periods=3).mean()

        high = data["high"].astype(float)
        low = data["low"].astype(float)
        spread = high - low
        candle_valid = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & spread.ge(0) & close.ge(low) & close.le(high)
        location = ((2.0 * close - high - low) / spread.where(spread > 0)).clip(-1.0, 1.0)
        out["close_location"] = location.where(spread.ne(0), 0.0).where(candle_valid)
        volume = data["volume"].astype(float)
        out["valid_volume"] = volume.where(np.isfinite(volume) & volume.ge(0) & candle_valid)
        out["weighted_pressure"] = out["close_location"] * out["valid_volume"]
        out["pressure_sum"] = out["weighted_pressure"].rolling(params.flow_window, min_periods=params.flow_window).sum()
        out["volume_sum"] = out["valid_volume"].rolling(params.flow_window, min_periods=params.flow_window).sum()
        out["flow"] = out["pressure_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: VisibilityValleyParams) -> SignalFrame:
        topology_agrees = indicators["topology"].ge(params.topology_threshold)
        pressure_agrees = indicators["flow"].ge(params.flow_threshold)
        ready = indicators[["topology", "flow"]].notna().all(axis=1)
        eligible = ready & topology_agrees & pressure_agrees
        df = pd.DataFrame(index=data.index)
        # The target stays long only while both entry primitives agree.
        # A false conjunction is the sole exit condition; no timer or price stop.
        df["signal"] = eligible.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityValleyParams()
    index = pd.date_range("2020-01-01", periods=60, freq="D")
    close = 100.0 + 3.0 * np.cos(np.arange(60) / 4.0)
    data = pd.DataFrame({"open": close - 0.2, "high": close + 0.4, "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:40], params)
    pd.testing.assert_frame_equal(full.iloc[:40], prefix)
    assert full["topology"].iloc[:params.graph_window + 2].isna().all()
    assert strategy.warmup_bars(params) >= params.graph_window + 3

    probe = pd.DataFrame({"topology": [np.nan, 0.3, 0.3, 0.0, 0.3, 0.3], "flow": [np.nan, 0.2, 0.2, 0.2, -0.2, 0.2]}, index=index[:6])
    result = strategy.generate_signals(data.iloc[:6], probe, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert result["signal"].isin([0, 1]).all()


if __name__ == "__main__":
    _self_check()
