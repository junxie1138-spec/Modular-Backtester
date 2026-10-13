from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeSeasonalityParams:
    expansion_ratio: float = 1.3
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[RangeSeasonalityParams]):
    strategy_id = "gen_local-w2_1791230634"

    @classmethod
    def params_type(cls):
        return RangeSeasonalityParams

    def warmup_bars(self, params):
        return 255

    def indicators(self, data, params):
        if not np.isfinite(params.expansion_ratio) or params.expansion_ratio <= 1:
            raise ValueError("expansion_ratio must be finite and greater than one")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        daily_range = (data["high"] - data["low"]).clip(lower=0)
        out["range"] = daily_range
        out["range_ratio"] = daily_range.div(daily_range.shift(1).replace(0, np.nan))
        true_range = pd.concat(
            [daily_range, (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        out["return"] = data["close"].div(previous_close.replace(0, np.nan)) - 1
        out["weekday"] = data.index.dayofweek
        out["expanded"] = out["range_ratio"].ge(params.expansion_ratio)
        out["seasonal_edge"] = np.nan
        out["seasonal_count"] = 0.0
        for weekday in range(5):
            # A sample at t pairs expansion at t-1 with the return ending at t.
            eligible = (
                out["expanded"].shift(1).eq(True)
                & out["weekday"].shift(1).eq(weekday)
                & np.isfinite(out["return"])
            )
            count = eligible.astype(float).rolling(252, min_periods=252).sum().shift(1)
            total = out["return"].where(eligible, 0.0).rolling(
                252, min_periods=252
            ).sum().shift(1)
            out[f"count_{weekday}"] = count
            out[f"edge_{weekday}"] = total.div(count.replace(0, np.nan))
            today = out["weekday"].eq(weekday)
            out.loc[today, "seasonal_count"] = count.loc[today]
            out.loc[today, "seasonal_edge"] = out.loc[today, f"edge_{weekday}"]
        valid = (
            out["expanded"] & out["seasonal_count"].ge(8)
            & out["seasonal_edge"].abs().gt(0.0005)
            & np.isfinite(out["atr"]) & out["atr"].gt(0)
        )
        out["entry"] = np.where(
            valid, np.sign(out["seasonal_edge"].fillna(0)), 0
        ).astype(int)
        out.iloc[:self.warmup_bars(params), out.columns.get_loc("entry")] = 0
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        stop = np.nan
        fixed_atr = np.nan
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # The preceding decision fills at this session's open.
                    stop = opens[i] - position * params.stop_atr * fixed_atr
                invalid = not np.isfinite(closes[i]) or not np.isfinite(stop)
                stopped = position * (closes[i] - stop) < 0
                expired = i - entry_bar + 1 >= 2
                if invalid or stopped or expired:
                    position = 0
                decisions[i] = position
                continue
            if entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                fixed_atr = atr[i]
                decisions[i] = position
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = RangeSeasonalityParams()
    index = pd.date_range("2020-01-06", periods=5, freq="B")
    bars = pd.DataFrame({"open": 100.0, "close": 100.0}, index=index)
    derived = pd.DataFrame({"entry": [1, 0, 0, 0, 0], "atr": 1.0}, index=index)
    assert strategy.generate_signals(bars, derived, None, params).data["signal"].tolist() == [0, 1, 1, 0, 0]
    for direction in (1, -1):
        derived["entry"] = [direction, 0, 0, 0, 0]
        bars["close"] = [100, 100 - direction * 2, 100, 100, 100]
        result = strategy.generate_signals(bars, derived, None, params).data
        assert result["signal"].tolist() == [0, direction, 0, 0, 0]
        assert (result["size"] > 0).all()
