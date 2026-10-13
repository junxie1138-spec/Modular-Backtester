from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_min: int = 2
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790907432"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.streak_min < 1 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("streak_min and trail_atr must be positive")
        return int(21 + params.streak_min)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        warmup = self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["previous_close"] = close.shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["prior_floor"] = data["low"].rolling(20, min_periods=20).min().shift(1)
        out["prior_ceiling"] = data["high"].rolling(20, min_periods=20).max().shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["fresh_low"] = data["low"] < out["prior_floor"]
        out["fresh_high"] = data["high"] > out["prior_ceiling"]
        for label in ("low", "high"):
            events = out["fresh_" + label]
            out[label + "_streak"] = events.astype(int).groupby((~events).cumsum()).cumsum()
            out["previous_" + label + "_streak"] = out[label + "_streak"].shift(1).fillna(0)
        denominator = out["atr"].where(out["atr"] > 0)
        out["drawdown_atr"] = (out["prior_ceiling"] - close) / denominator
        out["drawup_atr"] = (close - out["prior_floor"]) / denominator
        valid_prices = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        out["ready"] = (
            (np.arange(len(data)) >= warmup - 1)
            & valid_prices
            & (close > 0)
            & out["prior_floor"].notna()
            & out["prior_ceiling"].notna()
            & (out["atr"] > 0)
        )
        long_entry = (
            out["ready"]
            & (out["previous_low_streak"] >= params.streak_min)
            & ~out["fresh_low"]
            & (close > out["previous_high"])
            & (out["drawdown_atr"] >= 1.0)
        )
        short_entry = (
            out["ready"]
            & (out["previous_high_streak"] >= params.streak_min)
            & ~out["fresh_high"]
            & (close < out["previous_low"])
            & (out["drawup_atr"] >= 1.0)
        )
        out["candidate"] = long_entry.astype(int) - short_entry.astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        side = age = 0
        entry = stop = 0.0
        pending = False
        for i in range(len(data)):
            atr = float(indicators["atr"].iat[i])
            if side:
                if pending:
                    opening = float(data["open"].iat[i])
                    if not np.isfinite(opening) or opening <= 0:
                        side = age = 0
                        pending = False
                        continue
                    # Entry occurs on the open following the decision bar.
                    entry = side * opening
                    stop = entry - 2.0 * float(indicators["atr"].iat[i - 1])
                    pending = False
                age += 1
                adverse = float(data["low" if side == 1 else "high"].iat[i])
                closing = float(data["close"].iat[i])
                # Test the existing stop before updating it with this close.
                # A breach schedules a flat position for the next open.
                if (
                    not np.isfinite(adverse)
                    or not np.isfinite(closing)
                    or side * adverse <= stop
                    or age >= 4
                ):
                    side = age = 0
                else:
                    signed_close = side * closing
                    if signed_close >= entry + abs(entry) * 0.006:
                        stop = max(stop, entry)
                    if stop >= entry and np.isfinite(atr) and atr > 0:
                        # Signed-price stops only move up; short dollar stops move down.
                        stop = max(stop, signed_close - params.trail_atr * atr)
            elif bool(indicators["ready"].iat[i]):
                side = int(indicators["candidate"].iat[i])
                if side:
                    pending = True
                    age = 0
            df.iat[i, 0] = side
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100.0] * 7,
        "high": [100.3, 100.4, 101.2, 101.1, 100.3, 100.3, 100.3],
        "low": [99.8, 99.9, 100.0, 100.5, 99.8, 99.8, 99.8],
        "close": [100.0, 100.2, 101.0, 100.8, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 7,
    }, index=index)
    ind = pd.DataFrame({"atr": 0.2, "ready": True, "candidate": [1, 0, 0, 0, 0, 0, 0]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    mirrored = data.copy()
    mirrored["open"] = 200.0 - data["open"]
    mirrored["close"] = 200.0 - data["close"]
    mirrored["high"] = 200.0 - data["low"]
    mirrored["low"] = 200.0 - data["high"]
    ind["candidate"] = -ind["candidate"]
    assert strategy.generate_signals(mirrored, ind, ctx, params).data["signal"].tolist() == [0, -1, -1, -1, 0, 0, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    ind["candidate"] = -ind["candidate"]
    assert strategy.generate_signals(flat, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    warm = strategy.indicators(data, params)
    assert not warm.isna().any().any()
    assert not warm["ready"].any()


if __name__ == "__main__":
    _self_check()
