from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WeeklyVisibilityParams:
    min_analog_move: float = 0.002
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[WeeklyVisibilityParams]):
    strategy_id = "gen_local-w2_1791317333"

    @classmethod
    def params_type(cls):
        return WeeklyVisibilityParams

    def warmup_bars(self, params: WeeklyVisibilityParams) -> int:
        if not np.isfinite(params.min_analog_move) or params.min_analog_move < 0:
            raise ValueError("min_analog_move must be finite and nonnegative")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        return 9

    def indicators(self, data: pd.DataFrame, params: WeeklyVisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        ready = close.rolling(4, min_periods=4).count().eq(4)
        upper = pd.Series(1.0, index=data.index)
        lower = pd.Series(1.0, index=data.index)
        # Endpoint degrees in horizontal visibility graphs of price and inverted price.
        for lag in (2, 3):
            upper_edge = ready.copy()
            lower_edge = ready.copy()
            ceiling = np.minimum(close, close.shift(lag))
            floor = np.maximum(close, close.shift(lag))
            for intermediate in range(1, lag):
                upper_edge &= close.shift(intermediate).lt(ceiling)
                lower_edge &= close.shift(intermediate).gt(floor)
            upper += upper_edge.astype(float)
            lower += lower_edge.astype(float)
        out["upper_degree"] = upper.where(ready)
        out["lower_degree"] = lower.where(ready)
        out["prior_upper_degree"] = out["upper_degree"].shift(5)
        out["prior_lower_degree"] = out["lower_degree"].shift(5)
        opening = data["open"].where(np.isfinite(data["open"]) & data["open"].gt(0))
        out["analog_return"] = (close / opening - 1.0).shift(4)
        dates = pd.Series(data.index, index=data.index)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        out["weekly_alignment"] = dates.diff(5).dt.days.eq(7) & weekday.shift(4).eq((weekday + 1) % 5)
        out["motif_match"] = (
            ready
            & out["upper_degree"].eq(out["prior_upper_degree"])
            & out["lower_degree"].eq(out["prior_lower_degree"])
            & out["upper_degree"].ne(out["lower_degree"])
        )
        eligible = (
            out["weekly_alignment"]
            & out["motif_match"]
            & out["analog_return"].abs().ge(params.min_analog_move)
            & out["analog_return"].ne(0)
        )
        out["candidate"] = np.sign(out["analog_return"].fillna(0)).where(eligible, 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WeeklyVisibilityParams) -> SignalFrame:
        self.warmup_bars(params)
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        pending = 0
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            # Yesterday's decision fills at today's open.
            if pending == 0:
                position = 0
            elif position == 0:
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    pending = 0
                    continue
                position = pending
                entry_price = opening[i]
                held = 0
            if position:
                held += 1
                gain = position * (closing[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                # Close-observed exits execute at the following open.
                raw[i] = 0 if target_hit or held >= params.max_hold_bars else position
            else:
                raw[i] = candidate[i]
            pending = int(raw[i])
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2024-01-01", periods=30)
    close = np.tile([100.0, 103.0, 101.0, 99.0, 102.0], 6)
    data = pd.DataFrame({"open": close / 1.004, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000000.0}, index=index)
    strategy = GeneratedStrategy()
    params = WeeklyVisibilityParams(profit_target=1.0, max_hold_bars=1)
    features = strategy.indicators(data, params)
    result = strategy.generate_signals(data, features, None, params).data
    assert strategy.warmup_bars(params) == 9
    assert result["signal"].iloc[:9].eq(0).all()
    assert result["signal"].ne(0).any()
    assert not (result["signal"].ne(0) & result["signal"].shift(1).fillna(0).ne(0)).any()
    assert result["signal"].isin([-1, 0, 1]).all()
    assert result["size"].gt(0).all()
    prefix = data.iloc[:20]
    prefix_result = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
    assert result.iloc[:20].equals(prefix_result)
    target_params = WeeklyVisibilityParams(profit_target=0.001, max_hold_bars=2)
    target_result = strategy.generate_signals(data, features, None, target_params).data
    assert target_result["signal"].equals(result["signal"])
    short_data = data.copy()
    short_data["open"] = close / 0.996
    short_result = strategy.generate_signals(short_data, strategy.indicators(short_data, params), None, params).data
    assert short_result["signal"].equals(-result["signal"])


if __name__ == "__main__":
    _self_check()
