from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    baseline_bars: int = 20
    compression_ratio: float = 0.75
    min_streak: int = 3
    wick_fraction: float = 0.45
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791255773"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(params.baseline_bars + params.min_streak + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if (
            params.baseline_bars < 2 or params.min_streak < 2
            or not 0 < params.compression_ratio < 1
            or not 0.3 <= params.wick_fraction <= 0.8
            or not 0 < params.profit_target < 1
            or params.max_hold_bars not in (1, 2)
        ):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["relative_range"] = span / data["close"].where(data["close"] > 0)
        out["range_baseline"] = out["relative_range"].shift(1).rolling(
            params.baseline_bars, min_periods=params.baseline_bars
        ).mean()
        out["compressed"] = (
            out["relative_range"] <= params.compression_ratio * out["range_baseline"]
        ).fillna(False)
        out["compression_streak"] = out["compressed"].astype(int).groupby(
            (~out["compressed"]).cumsum()
        ).cumsum()
        out["upper_wick"] = (data["high"] - data[["open", "close"]].max(axis=1)) / span
        out["lower_wick"] = (data[["open", "close"]].min(axis=1) - data["low"]) / span
        out["close_location"] = (data["close"] - data["low"]) / span
        valid = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        out["rejection"] = np.select(
            [
                valid & out["compressed"] & (out["lower_wick"] >= params.wick_fraction)
                & (out["close_location"] >= 0.6),
                valid & out["compressed"] & (out["upper_wick"] >= params.wick_fraction)
                & (out["close_location"] <= 0.4),
            ],
            [1, -1], default=0,
        )
        out["confirmed"] = out["rejection"].where(
            out["rejection"].eq(out["rejection"].shift(1))
            & (out["compression_streak"] >= params.min_streak), 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        candidates = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # The previous close's entry decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = side * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                # Closing-price target; exit decision fills at the following open.
                if (
                    not valid_entry or gain >= params.profit_target
                    or i - entry_bar + 1 >= params.max_hold_bars
                ):
                    side = 0
            elif i >= self.warmup_bars(params) - 1 and candidates[i]:
                if np.isfinite(closes[i]) and closes[i] > 0:
                    side = int(candidates[i])
                    entry_bar = i + 1
                    entry_price = np.nan
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(baseline_bars=2, min_streak=2)
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame({
        "open": [100.0] * 9, "high": [104.0] * 9,
        "low": [96.0] * 9, "close": [100.0] * 9, "volume": [1000.0] * 9,
    }, index=index)
    strategy = GeneratedStrategy()
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    for side in (1, -1):
        derived = pd.DataFrame({"confirmed": [0, 0, 0, 0, side, 0, 0, 0, 0]}, index=index)
        signals = strategy.generate_signals(data, derived, ctx, params).data
        assert signals["signal"].tolist() == [0, 0, 0, 0, 0, side, side, 0, 0]
        profitable = data.copy()
        profitable.iloc[5, profitable.columns.get_loc("close")] = 100.0 * (1 + side * 0.01)
        signals = strategy.generate_signals(profitable, derived, ctx, params).data
        assert signals["signal"].tolist() == [0, 0, 0, 0, 0, side, 0, 0, 0]
    data.iloc[4:, data.columns.get_loc("high")] = 101.0
    data.iloc[4:, data.columns.get_loc("low")] = 99.0
    data.iloc[4:, data.columns.get_loc("open")] = 100.4
    data.iloc[4:, data.columns.get_loc("close")] = 100.6
    derived = strategy.indicators(data, params)
    assert derived["confirmed"].iloc[4] == 0
    assert derived["confirmed"].iloc[5] == 1
    assert strategy.indicators(data.iloc[:1], params)["confirmed"].eq(0).all()


if __name__ == "__main__":
    _self_check()
