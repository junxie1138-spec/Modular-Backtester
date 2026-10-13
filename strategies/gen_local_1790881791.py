from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AuctionParams:
    streak_bars: int = 4
    slope_bars: int = 20
    slope_threshold: float = 0.6
    atr_bars: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[AuctionParams]):
    strategy_id = "gen_local_1790881791"

    @classmethod
    def params_type(cls):
        return AuctionParams

    def warmup_bars(self, params: AuctionParams) -> int:
        return int(max(200 + params.slope_bars, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: AuctionParams) -> pd.DataFrame:
        if min(params.streak_bars, params.slope_bars, params.atr_bars, params.hold_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not all(np.isfinite(x) and x > 0 for x in (params.slope_threshold, params.breakeven_pct, params.trail_atr)):
            raise ValueError("Thresholds and stop parameters must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        prior_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        tr = tr.where(prior_close.notna())
        out["atr"] = tr.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["slope"] = out["ma200"].diff(params.slope_bars) / out["atr"].replace(0.0, np.nan)
        out["prior_midpoint"] = ((data["high"] + data["low"]) * 0.5).shift(1)
        up = (close > out["prior_midpoint"]).fillna(False)
        down = (close < out["prior_midpoint"]).fillna(False)
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()
        out["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum()
        ready = np.isfinite(out[["ma200", "atr", "slope"]]).all(axis=1) & out["atr"].gt(0)
        flat = out["slope"].abs().lt(params.slope_threshold)
        bull = close.gt(out["ma200"])
        bear = close.lt(out["ma200"])
        long_setup = bull & (
            (out["slope"].ge(params.slope_threshold) & out["up_streak"].eq(params.streak_bars))
            | (flat & out["down_streak"].eq(params.streak_bars))
        )
        short_setup = bear & (
            (out["slope"].le(-params.slope_threshold) & out["down_streak"].eq(params.streak_bars))
            | (flat & out["up_streak"].eq(params.streak_bars))
        )
        out["candidate"] = np.select([ready & long_setup, ready & short_setup], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AuctionParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        averages = indicators["ma200"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        side = 0
        pending = 0
        entry = 0.0
        entry_bar = 0
        favorable = 0.0
        stop = 0.0
        armed = False
        for i in range(len(data)):
            if pending:
                if i > 0 and np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atrs[i - 1]) and atrs[i - 1] > 0:
                    side = pending
                    entry = opens[i]
                    entry_bar = i
                    favorable = side * entry
                    stop = favorable - params.trail_atr * atrs[i - 1]
                    armed = False
                pending = 0
            valid = (
                np.isfinite([opens[i], highs[i], lows[i], closes[i], atrs[i], averages[i]]).all()
                and min(opens[i], highs[i], lows[i], closes[i], atrs[i]) > 0
            )
            if not valid:
                side = 0
                continue
            if side:
                # Signed prices make both long and short stop levels ratchet upward.
                adverse = lows[i] if side == 1 else -highs[i]
                exit_now = adverse <= stop
                if not exit_now:
                    extreme = highs[i] if side == 1 else -lows[i]
                    favorable = max(favorable, extreme)
                    if favorable - side * entry >= entry * params.breakeven_pct:
                        armed = True
                    if armed:
                        stop = max(stop, side * entry, favorable - params.trail_atr * atrs[i])
                        # A newly tightened stop is checked at the close, without assuming intrabar ordering.
                        exit_now = side * closes[i] <= stop
                exit_now = exit_now or i - entry_bar + 1 >= params.hold_bars
                exit_now = exit_now or side * (closes[i] - averages[i]) <= 0
                if exit_now:
                    side = 0
                    continue
                decisions[i] = side
            elif candidates[i] != 0:
                pending = int(candidates[i])
                decisions[i] = pending
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = AuctionParams(breakeven_pct=0.02, trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        close = np.array([100.0, 102.5, 100.5, 100.5, 100.5])
        high = np.array([100.5, 103.0, 101.0, 101.0, 101.0])
        low = np.array([99.5, 99.5, 100.0, 100.0, 100.0])
        if direction == -1:
            close, high, low = 200.0 - close, 200.0 - low, 200.0 - high
        data = pd.DataFrame({"open": 100.0, "high": high, "low": low, "close": close, "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"atr": 1.0, "ma200": 90.0 if direction == 1 else 110.0, "candidate": [direction, 0, 0, 0, 0]}, index=index)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, 0, 0]
        assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 200 + params.slope_bars


if __name__ == "__main__":
    _self_check()
