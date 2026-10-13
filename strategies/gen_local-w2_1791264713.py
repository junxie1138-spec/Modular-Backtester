from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_min: int = 3
    atr_period: int = 14
    breakeven_pct: float = 0.006
    trail_atr: float = 1.2
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791264713"

    @classmethod
    def params_type(cls):
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(params.atr_period + 1, params.streak_min + 3)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if (params.streak_min < 2 or params.atr_period < 2
                or not 3 <= params.max_hold <= 5
                or not np.isfinite(params.breakeven_pct)
                or not np.isfinite(params.trail_atr)
                or params.breakeven_pct <= 0 or params.trail_atr <= 0):
            raise ValueError("Invalid streak, ATR, activation, trail, or holding parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["change"] = data["close"].diff()
        down = out["change"].lt(0)
        up = out["change"].gt(0)
        out["down_streak"] = down.astype(int).groupby((~down).cumsum()).cumsum()
        out["up_streak"] = up.astype(int).groupby((~up).cumsum()).cumsum()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        # Both reversal closes must remain inside the terminal streak candle.
        out["long_confirmed"] = (
            out["down_streak"].shift(2).ge(params.streak_min)
            & out["change"].shift(1).gt(0) & out["change"].gt(0)
            & data["close"].le(data["high"].shift(2))
        ).fillna(False)
        out["short_confirmed"] = (
            out["up_streak"].shift(2).ge(params.streak_min)
            & out["change"].shift(1).lt(0) & out["change"].lt(0)
            & data["close"].ge(data["low"].shift(2))
        ).fillna(False)
        out["candidate"] = np.select(
            [out["long_confirmed"], out["short_confirmed"]], [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StreakParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidate = indicators["candidate"].to_numpy(dtype=int)
        valid = (
            np.isfinite(data[["open", "high", "low", "close"]].to_numpy(dtype=float)).all(axis=1)
            & (data[["open", "high", "low", "close"]].to_numpy(dtype=float) > 0).all(axis=1)
            & data["high"].ge(data["low"]).to_numpy()
            & np.isfinite(atr) & (atr > 0)
        )
        signals = np.zeros(len(data), dtype=int)
        side, held = 0, 0
        entry, peak, stop = np.nan, 0.0, 0.0
        armed = False
        for i in range(len(data)):
            if not valid[i]:
                side, entry = 0, np.nan
            elif side:
                if np.isnan(entry):
                    # The preceding close's decision enters at this bar's open.
                    entry = opens[i]
                    peak = side * entry
                    stop = peak - params.trail_atr * atr[i - 1]
                    held, armed = 0, False
                held += 1
                # Signed prices make the stop ratchet upward for either direction.
                signed_close = side * close[i]
                peak = max(peak, signed_close)
                armed = armed or side * (close[i] - entry) / entry >= params.breakeven_pct
                if armed:
                    stop = max(stop, side * entry, peak - params.trail_atr * atr[i])
                # Stops are evaluated at the close; exits fill on the following bar.
                if signed_close <= stop or held >= params.max_hold:
                    side, entry = 0, np.nan
            elif i >= self.warmup_bars(params) - 1 and candidate[i]:
                side, entry = int(candidate[i]), np.nan
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StreakParams(atr_period=3, breakeven_pct=0.5)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    close = pd.Series([100, 99, 98, 97, 97.3, 97.6, 98, 98.4, 98.5, 98.4, 98.2],
                      index=pd.date_range("2020-01-01", periods=11))
    bars = pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]),
                         "high": close + 0.8, "low": close - 0.8,
                         "close": close, "volume": 1000.0})
    result = strategy.generate_signals(bars, strategy.indicators(bars, params), ctx, params).data
    assert result["signal"].tolist() == [0] * 6 + [1] * 4 + [0]
    assert result["size"].gt(0).all()
    mirror = bars.copy()
    mirror["open"], mirror["close"] = 200 - bars["open"], 200 - bars["close"]
    mirror["high"], mirror["low"] = 200 - bars["low"], 200 - bars["high"]
    short = strategy.generate_signals(mirror, strategy.indicators(mirror, params), ctx, params).data
    assert short["signal"].tolist() == (-result["signal"]).tolist()
    params.breakeven_pct = 0.004
    reversal = bars.copy()
    reversal.loc[reversal.index[6], "close"] = 98.4
    reversal.loc[reversal.index[7], "close"] = 97.4
    reversal["high"], reversal["low"] = reversal["close"] + 0.8, reversal["close"] - 0.8
    stopped = strategy.generate_signals(reversal, strategy.indicators(reversal, params), ctx, params).data
    assert stopped["signal"].iloc[6:9].tolist() == [1, 1, 0]
    for n in range(1, len(reversal) + 1):
        prefix = reversal.iloc[:n]
        actual = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
        assert actual["signal"].equals(stopped["signal"].iloc[:n])
    params.trail_atr, params.max_hold = 0.3, 5
    trailing = bars.copy()
    trailing.loc[trailing.index[6:9], "close"] = [99.0, 100.0, 99.3]
    trailing["high"], trailing["low"] = trailing["close"] + 0.8, trailing["close"] - 0.8
    trailing.loc[trailing.index[8], ["high", "low"]] = [120.0, 80.0]
    ratchet = strategy.generate_signals(trailing, strategy.indicators(trailing, params), ctx, params).data
    assert ratchet["signal"].iloc[6:10].tolist() == [1, 1, 1, 0]
