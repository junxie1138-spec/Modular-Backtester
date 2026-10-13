from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_window: int = 12
    entry_threshold: float = 0.003


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790919751"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        # One return observation needs a preceding close; labels mature
        # after 18 sessions, followed by a full 504-session estimation window.
        return int(params.graph_window + 18 + 504)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.graph_window < 2 or not np.isfinite(params.entry_threshold) or params.entry_threshold <= 0:
            raise ValueError("graph_window must be >= 2 and entry_threshold must be finite and positive")

        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_return"] = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
        out["shock"] = out["log_return"].abs()
        shock = out["shock"].to_numpy()
        degree = np.zeros(len(data), dtype=float)
        barrier = np.full(len(data), -np.inf)

        # Endpoint horizontal visibility: both endpoints must exceed every
        # intervening shock. All comparisons use current or earlier bars.
        # ponytail: O(n * graph_window); record-stack traversal if windows grow large.
        for lag in range(1, params.graph_window):
            past = out["shock"].shift(lag).to_numpy()
            degree += (np.minimum(shock, past) > barrier).astype(float)
            barrier = np.maximum(barrier, past)

        out["graph_observations"] = out["shock"].rolling(params.graph_window).count()
        complete = out["graph_observations"].eq(params.graph_window)
        out["visibility_degree"] = pd.Series(degree, index=data.index).where(complete)
        # Three or more visible predecessors identify a connected shock state.
        out["topology_state"] = out["visibility_degree"].ge(3).astype(float).where(complete)
        out["calendar_month"] = pd.Series(data.index.month, index=data.index, dtype=float)

        # A label observed today belongs to the feature state 18 sessions ago.
        out["matured_return"] = (close / close.shift(18) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["matured_month"] = out["calendar_month"].shift(18)
        out["matured_state"] = out["topology_state"].shift(18)
        out["forecast"] = np.nan
        out["sample_count"] = np.nan

        for month in range(1, 13):
            for state in range(2):
                historical = (
                    out["matured_month"].eq(month)
                    & out["matured_state"].eq(state)
                    & out["matured_return"].notna()
                )
                prefix = f"month_{month}_state_{state}"
                out[f"{prefix}_count"] = historical.astype(float).rolling(504, min_periods=504).sum()
                out[f"{prefix}_sum"] = out["matured_return"].where(historical, 0.0).rolling(504, min_periods=504).sum()
                count = out[f"{prefix}_count"]
                out[f"{prefix}_mean"] = out[f"{prefix}_sum"] / count.where(count.ge(8))
                current = out["calendar_month"].eq(month) & out["topology_state"].eq(state)
                out.loc[current, "forecast"] = out.loc[current, f"{prefix}_mean"]
                out.loc[current, "sample_count"] = count.loc[current]

        out.loc[out.index[:self.warmup_bars(params) - 1], "forecast"] = np.nan
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        forecast = indicators["forecast"].to_numpy(dtype=float)
        candidate = np.zeros(len(data), dtype=int)
        candidate[np.isfinite(forecast) & (forecast >= params.entry_threshold)] = 1
        candidate[np.isfinite(forecast) & (forecast <= -params.entry_threshold)] = -1
        raw = np.zeros(len(data), dtype=int)
        position = 0

        for i, direction in enumerate(candidate):
            if position == 0:
                position = int(direction)
            elif direction == -position:
                # Opposite entry condition exits to flat. An opposite entry
                # can occur on a subsequent decision bar if it persists.
                position = 0
            raw[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.bdate_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    forecasts = pd.DataFrame(
        {"forecast": [0.006, np.nan, -0.006, -0.006, 0.0, 0.006, 0.006]},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, forecasts, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()

    graph_params = GeneratedParams(graph_window=4)
    graph_index = pd.bdate_range("2020-01-01", periods=6)
    graph_close = 100.0 * np.exp(np.cumsum([0.0, 0.01, 0.03, 0.02, 0.04, 0.005]))
    graph_data = pd.DataFrame({"close": graph_close}, index=graph_index)
    graph_out = strategy.indicators(graph_data, graph_params)
    assert graph_out["visibility_degree"].iloc[:4].isna().all()
    assert graph_out["visibility_degree"].iloc[4:].tolist() == [2.0, 1.0]

    index = pd.bdate_range("2018-01-01", periods=650)
    close = 100.0 * np.exp(np.cumsum(0.0003 + 0.008 * np.sin(np.arange(650) * 0.71)))
    data = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    full = strategy.indicators(data, params)
    prefix = strategy.indicators(data.iloc[:600], params)
    pd.testing.assert_frame_equal(full.iloc[:600], prefix)
    assert full["forecast"].iloc[:strategy.warmup_bars(params) - 1].isna().all()
    signals = strategy.generate_signals(data, full, ctx, params).data
    assert signals.index.equals(data.index)
    assert signals["signal"].isin([-1, 0, 1]).all()


if __name__ == "__main__":
    _self_check()
