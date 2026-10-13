from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalStreakParams:
    streak_months: int = 3
    atr_period: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18
    entry_session_limit: int = 6


class GeneratedStrategy(BaseStrategy[SeasonalStreakParams]):
    strategy_id = "gen_local_1791277277"

    @classmethod
    def params_type(cls):
        return SeasonalStreakParams

    def warmup_bars(self, params):
        if (
            params.streak_months < 1
            or params.atr_period < 1
            or params.hold_bars < 1
            or params.entry_session_limit < 4
            or not np.isfinite(params.stop_atr)
            or params.stop_atr <= 0
        ):
            raise ValueError("Invalid seasonal streak parameters")
        # At most 23 weekday sessions per month, plus an incomplete first month.
        return int(max(params.atr_period + 1, 23 * (params.streak_months + 1) + 4))

    def indicators(self, data, params):
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        month = data.index.to_period("M")
        out["month_session"] = data.groupby(month).cumcount() + 1
        out["month_open"] = (
            data["open"].where(out["month_session"].eq(1)).groupby(month).ffill()
        )
        out["opening_return"] = (
            data["close"].div(out["month_open"].where(out["month_open"].gt(0))) - 1.0
        ).replace([np.inf, -np.inf], np.nan)
        events = np.sign(out.loc[out["month_session"].eq(3), "opening_return"]).fillna(0).astype(int)
        runs = events.ne(events.shift()).cumsum()
        streak = (events.groupby(runs).cumcount() + 1).where(events.ne(0), 0)
        streak = streak.clip(upper=params.streak_months)
        out["monthly_direction"] = events.reindex(data.index).ffill().fillna(0).astype(int)
        out["monthly_streak"] = streak.reindex(data.index).ffill().fillna(0).astype(int)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1).where(data[["high", "low", "close"]].notna().all(axis=1))
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["close_change"] = data["close"].diff()
        eligible = (
            out["month_session"].between(3, params.entry_session_limit)
            & out["monthly_streak"].ge(params.streak_months)
            & (out["close_change"] * out["monthly_direction"]).gt(0)
            & out["atr"].gt(0)
            & np.isfinite(out["atr"])
        )
        out["candidate"] = out["monthly_direction"].where(eligible, 0).astype(int)
        confirmed = (
            out["candidate"].ne(0)
            & out["candidate"].eq(out["candidate"].shift(1))
            & out["month_session"].gt(3)
        )
        out["confirmed"] = out["candidate"].where(confirmed, 0).astype(int)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        warmup = self.warmup_bars(params)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        decision_bar = -1
        entry_atr = np.nan
        stop = np.nan

        for i in range(len(data)):
            if position:
                if i == decision_bar + 1:
                    # The order decided yesterday fills at today's open.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                invalid = (
                    not np.isfinite(closes[i])
                    or closes[i] <= 0
                    or not np.isfinite(stop)
                    or (i == decision_bar + 1 and (not np.isfinite(opens[i]) or opens[i] <= 0))
                )
                stopped = position * (closes[i] - stop) < 0
                expired = i - decision_bar >= params.hold_bars
                if invalid or stopped or expired:
                    position = 0
            elif (
                i >= warmup - 1
                and confirmed[i] != 0
                and np.isfinite(closes[i])
                and closes[i] > 0
                and np.isfinite(atr[i])
                and atr[i] > 0
            ):
                position = int(confirmed[i])
                decision_bar = i
                entry_atr = atr[i]
                stop = np.nan
            signal[i] = position

        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SeasonalStreakParams()
    dates = pd.bdate_range("2020-01-01", periods=260)
    month = dates.to_period("M")
    session = pd.Series(1, index=dates).groupby(month).cumcount() + 1
    direction = np.where(dates.month <= 6, 1, -1)
    close = 100.0 + direction * session.clip(upper=6).to_numpy() * 0.2
    data = pd.DataFrame(
        {
            "open": 100.0,
            "high": np.maximum(100.0, close) + 1.0,
            "low": np.minimum(100.0, close) - 1.0,
            "close": close,
            "volume": 1000000.0,
        },
        index=dates,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = strategy.indicators(data, params)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert set(result["signal"].unique()) == {-1, 0, 1}
    assert result["size"].gt(0).all()
    assert indicators.loc[indicators["confirmed"].ne(0), "month_session"].ge(4).all()
    prefix = data.iloc[:180]
    prefix_result = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), ctx, params
    ).data
    assert result.iloc[:180].equals(prefix_result)

    params = SeasonalStreakParams(streak_months=1, stop_atr=2.0, hold_bars=3)
    decision = strategy.warmup_bars(params) - 1
    flat = data.iloc[:decision + 8].copy()
    flat[["open", "close"]] = 100.0
    manual = pd.DataFrame({"atr": 1.0, "confirmed": 0}, index=flat.index)
    for side in (1, -1):
        manual.iloc[decision, manual.columns.get_loc("confirmed")] = side
        timed = strategy.generate_signals(flat, manual, ctx, params).data["signal"]
        assert timed.iloc[decision] == 0
        assert timed.iloc[decision + 1:decision + 4].eq(side).all()
        assert timed.iloc[decision + 4] == 0
        gapped = flat.copy()
        fill = 100.0 + 10.0 * side
        gapped.iloc[decision + 1, gapped.columns.get_loc("open")] = fill
        gapped.iloc[decision + 1, gapped.columns.get_loc("close")] = fill - side
        gapped.iloc[decision + 2, gapped.columns.get_loc("close")] = fill - 3.0 * side
        stopped = strategy.generate_signals(gapped, manual, ctx, params).data["signal"]
        assert stopped.iloc[decision + 1:decision + 3].eq(side).all()
        assert stopped.iloc[decision + 3] == 0


if __name__ == "__main__":
    _self_check()
