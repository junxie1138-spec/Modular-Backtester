from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarAreaParams:
    path_bars: int = 8
    smooth_bars: int = 3
    rank_bars: int = 126
    percentile: float = 0.60
    boundary_days: int = 7


class GeneratedStrategy(BaseStrategy[CalendarAreaParams]):
    strategy_id = "gen_local-w2_1791274129"

    @classmethod
    def params_type(cls):
        return CalendarAreaParams

    def warmup_bars(self, params: CalendarAreaParams) -> int:
        return int(params.path_bars + params.smooth_bars + params.rank_bars + 1)

    def indicators(self, data: pd.DataFrame, params: CalendarAreaParams) -> pd.DataFrame:
        if min(params.path_bars, params.smooth_bars, params.rank_bars) < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 0.0 < params.percentile < 1.0 or not 1 <= params.boundary_days <= 10:
            raise ValueError("Invalid percentile or month-boundary window.")
        out = pd.DataFrame(index=data.index)
        day = pd.Series(data.index.day, index=data.index)
        days = pd.Series(data.index.days_in_month, index=data.index)
        out["phase"] = (day - 1.0) / days
        out["calendar"] = np.cos(2.0 * np.pi * out["phase"])
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        x, y = out["log_close"], out["calendar"]
        length = params.path_bars
        out["segment_cross"] = x.shift(1) * y - y.shift(1) * x
        out["price_displacement"] = x - x.shift(length)
        out["calendar_displacement"] = y - y.shift(length)
        # Antisymmetric level-two signature of the translated open path.
        out["area"] = 0.5 * (
            out["segment_cross"].rolling(length, min_periods=length).sum()
            - x.shift(length) * out["calendar_displacement"]
            + y.shift(length) * out["price_displacement"]
        )
        out["score"] = out["area"].rolling(
            params.smooth_bars, min_periods=params.smooth_bars
        ).mean()
        out["threshold"] = out["score"].shift(1).rolling(
            params.rank_bars, min_periods=params.rank_bars
        ).quantile(params.percentile)
        out["season"] = (day <= params.boundary_days) | (
            days - day < params.boundary_days
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CalendarAreaParams,
    ) -> SignalFrame:
        valid = indicators[["score", "threshold"]].notna().all(axis=1)
        qualified = valid & indicators["season"].fillna(False) & (
            indicators["score"] > indicators["threshold"]
        )
        df = pd.DataFrame(index=data.index)
        # Qualification reversal is the sole exit; no time or price stops.
        df["signal"] = qualified.fillna(False).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = CalendarAreaParams()
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=420)
    t = np.arange(len(index))
    phase = (index.day.to_numpy() - 1.0) / index.days_in_month.to_numpy()
    returns = 0.001 + 0.004 * np.sin(2 * np.pi * phase + 0.8) + 0.003 * np.sin(t * 0.31)
    close = 100 * np.exp(np.cumsum(returns))
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1000000.0}, index=index
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    signals = strategy.generate_signals(data, ind, ctx, params).data
    expected = (ind["season"] & (ind["score"] > ind["threshold"])).astype(int)
    pd.testing.assert_series_equal(
        signals["signal"], expected.shift(1).fillna(0).astype(int),
        check_names=False
    )
    cut = 300
    prefix = strategy.indicators(data.iloc[:cut], params)
    pd.testing.assert_frame_equal(prefix, ind.iloc[:cut])
    x = ind["log_close"].iloc[-params.path_bars - 1:].to_numpy()
    y = ind["calendar"].iloc[-params.path_bars - 1:].to_numpy()
    dx, dy = np.diff(x), np.diff(y)
    area = 0.5 * sum(
        dx[i] * dy[j] - dy[i] * dx[j]
        for i in range(len(dx)) for j in range(i + 1, len(dx))
    )
    assert np.isclose(ind["area"].iloc[-1], area, atol=1e-12)
    assert signals["signal"].isin([0, 1]).all() and (signals["size"] > 0).all()
    assert signals["signal"].sum() > 0
    assert signals["signal"].diff().eq(-1).any()
    assert signals["signal"].iloc[:strategy.warmup_bars(params) - 1].eq(0).all()


if __name__ == "__main__":
    _self_check()
