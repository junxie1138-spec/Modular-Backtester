from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarAreaParams:
    path_bars: int = 21
    smooth_bars: int = 18
    percentile_bars: int = 252
    entry_percentile: float = 0.65


class GeneratedStrategy(BaseStrategy[CalendarAreaParams]):
    strategy_id = "gen_local-w2_1791214040"

    @classmethod
    def params_type(cls) -> type[CalendarAreaParams]:
        return CalendarAreaParams

    @staticmethod
    def warmup_bars(params: CalendarAreaParams) -> int:
        return int(params.path_bars + params.smooth_bars + params.percentile_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: CalendarAreaParams) -> pd.DataFrame:
        if min(params.path_bars, params.smooth_bars, params.percentile_bars) < 2:
            raise ValueError("Window lengths must be at least two.")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one.")
        out = pd.DataFrame(index=data.index)
        out["month_phase"] = (data.index.day - 1) / data.index.days_in_month
        out["calendar_pressure"] = -np.cos(2.0 * np.pi * out["month_phase"])
        out["log_price"] = np.log(data["close"].where(data["close"] > 0))
        x = out["calendar_pressure"]
        y = out["log_price"]
        out["calendar_increment"] = x.diff()
        out["price_increment"] = y.diff()
        dx = out["calendar_increment"]
        dy = out["price_increment"]
        out["cross_increment"] = x.shift(1) * dy - y.shift(1) * dx
        length = params.path_bars
        # Translate each window's path origin before computing its second-order area.
        out["signature_area"] = 0.5 * (
            out["cross_increment"].rolling(length, min_periods=length).sum()
            - x.shift(length) * (y - y.shift(length))
            + y.shift(length) * (x - x.shift(length))
        )
        out["path_scale"] = (
            dx.abs().rolling(length, min_periods=length).sum()
            * dy.abs().rolling(length, min_periods=length).sum()
        )
        scale = out["path_scale"]
        out["normalized_area"] = out["signature_area"] / scale.where(scale > 0)
        out.loc[scale.eq(0), "normalized_area"] = 0.0
        out["area_score"] = out["normalized_area"].rolling(
            params.smooth_bars, min_periods=params.smooth_bars
        ).mean()
        out["entry_threshold"] = out["area_score"].shift(1).rolling(
            params.percentile_bars, min_periods=params.percentile_bars
        ).quantile(params.entry_percentile)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CalendarAreaParams,
    ) -> SignalFrame:
        score = indicators["area_score"]
        threshold = indicators["entry_threshold"]
        entry_condition = score.notna() & threshold.notna() & score.gt(threshold)
        df = pd.DataFrame(index=data.index)
        # The same predicate holds the position and exits when it flips false.
        df["signal"] = entry_condition.astype(int)
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = CalendarAreaParams()
    index = pd.bdate_range("2015-01-02", periods=800)
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.0003 * t + 0.025 * np.sin(t / 13.0))
    data = pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": np.full(len(index), 1000000.0),
    }, index=index)
    derived = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:600], params)
    pd.testing.assert_frame_equal(derived.iloc[:600], prefix)
    length = params.path_bars
    x = derived["calendar_pressure"].iloc[-length - 1:].to_numpy()
    y = derived["log_price"].iloc[-length - 1:].to_numpy()
    direct = 0.5 * np.sum(
        (x[:-1] - x[0]) * np.diff(y)
        - (y[:-1] - y[0]) * np.diff(x)
    )
    assert np.isclose(derived["signature_area"].iloc[-1], direct, atol=1e-12)
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    expected = derived["area_score"].gt(derived["entry_threshold"]).astype(int)
    expected = expected.shift(1).fillna(0).astype(int)
    assert np.array_equal(result["signal"].to_numpy(), expected.to_numpy())
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0).all()
    assert derived["entry_threshold"].iloc[GeneratedStrategy.warmup_bars(params):].notna().all()


if __name__ == "__main__":
    _self_check()
