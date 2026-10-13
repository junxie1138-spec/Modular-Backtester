from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalBudgetParams:
    residual_z: float = 0.8
    trail_atr: float = 1.5

    def __post_init__(self):
        if not np.isfinite(self.residual_z) or self.residual_z <= 0:
            raise ValueError("residual_z must be positive and finite")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")


class GeneratedStrategy(BaseStrategy[SeasonalBudgetParams]):
    strategy_id = "gen_local_1791261719"

    @classmethod
    def params_type(cls):
        return SeasonalBudgetParams

    def warmup_bars(self, params):
        return 254

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["volatility"] = out["return"].shift(1).rolling(252, min_periods=252).std()
        weekday = data.index.dayofweek
        out["expected_daily"] = 0.0
        for day in range(5):
            sample = out["return"].where(weekday == day).shift(1)
            out[f"weekday_{day}"] = sample.rolling(252, min_periods=20).mean()
            out.loc[weekday == day, "expected_daily"] = out.loc[weekday == day, f"weekday_{day}"]
        week = pd.Series(data.index.to_period("W-SUN"), index=data.index)
        out["week_return"] = out["return"].groupby(week).cumsum()
        out["week_budget"] = out["expected_daily"].groupby(week).cumsum()
        out["week_sessions"] = data["close"].groupby(week).cumcount() + 1
        denominator = out["volatility"] * np.sqrt(out["week_sessions"])
        out["residual"] = (out["week_return"] - out["week_budget"]) / denominator.replace(0.0, np.nan)
        previous = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        ready = out[[f"weekday_{day}" for day in range(5)]].notna().all(axis=1)
        eligible = ready & (weekday == 3) & np.isfinite(out["residual"]) & (out["atr"] > 0)
        out["candidate"] = np.select([
            eligible & (out["residual"] < -params.residual_z),
            eligible & (out["residual"] > params.residual_z),
        ], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        raw = np.zeros(len(data), dtype=int)
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side = 0
        fill_bar = -1
        entry = stop = 0.0
        armed = False
        for i in range(len(data)):
            if side:
                if i == fill_bar:
                    if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        side = 0
                        continue
                    entry = opens[i]
                    stop = side * entry - 2.0 * atr[i - 1]
                    armed = False
                if not np.isfinite(closes[i]) or closes[i] <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                    side = 0
                    continue
                oriented_close = side * closes[i]
                # Stops are observed at closes; exits execute on the next bar.
                if oriented_close <= stop or i - fill_bar + 1 >= 4:
                    side = 0
                else:
                    if side * (closes[i] / entry - 1.0) >= 0.0075:
                        armed = True
                    if armed:
                        # Signed-price stops move only upward; short price stops move downward.
                        stop = max(stop, side * entry, oriented_close - params.trail_atr * atr[i])
                    raw[i] = side
            elif candidate[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(candidate[i])
                fill_bar = i + 1
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-06", periods=12)
    strategy = GeneratedStrategy()
    params = SeasonalBudgetParams()
    for side in (1, -1):
        data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
        ind = pd.DataFrame({"candidate": 0, "atr": 0.5}, index=index)
        ind.iloc[3, 0] = side
        held = strategy.generate_signals(data, ind, None, params).data["signal"]
        assert held.iloc[:4].eq(0).all()
        assert held.iloc[4:8].eq(side).all() and held.iloc[8:].eq(0).all()
        data.iloc[5, data.columns.get_loc("close")] = 100.0 + side
        data.iloc[6, data.columns.get_loc("close")] = 100.0 + 0.1 * side
        ind.iloc[6, 1] = 2.0
        trailed = strategy.generate_signals(data, ind, None, params).data["signal"]
        assert trailed.iloc[4:7].eq(side).all() and trailed.iloc[7:].eq(0).all()


if __name__ == "__main__":
    _self_check()
