from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_window: int = 8
    rank_window: int = 126
    rejection_window: int = 4
    tail: float = 0.10
    rejection_atr: float = 0.5
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791249388"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(params.gap_window + params.rank_window,
                       params.rejection_window, params.atr_window + 1))

    def indicators(self, data, params):
        previous = data["close"].shift(1).replace(0.0, np.nan)
        opening = data["open"].replace(0.0, np.nan)
        gap = data["open"].div(previous).sub(1.0)
        intraday = data["close"].div(opening).sub(1.0)
        gap_sum = gap.rolling(params.gap_window).sum()
        rejection = intraday.rolling(params.rejection_window).sum()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window).mean()
        rank = gap_sum.rolling(params.rank_window).rank(pct=True)
        threshold = params.rejection_atr * atr.div(data["close"].replace(0.0, np.nan))
        out = pd.DataFrame({
            "gap": gap, "intraday": intraday, "gap_sum": gap_sum,
            "rejection": rejection, "true_range": true_range,
            "atr": atr, "gap_rank": rank, "rejection_threshold": threshold
        }, index=data.index).replace([np.inf, -np.inf], np.nan)
        ready = out[["gap_sum", "gap_rank", "rejection", "atr",
                     "rejection_threshold"]].notna().all(axis=1) & atr.gt(0)
        out["long_entry"] = (ready & gap_sum.lt(0) & rank.le(params.tail)
                             & rejection.gt(threshold)).fillna(False)
        out["short_entry"] = (ready & gap_sum.gt(0) & rank.ge(1.0 - params.tail)
                              & rejection.lt(-threshold)).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        if (min(params.gap_window, params.rank_window, params.rejection_window,
                params.atr_window, params.max_hold) < 1
                or not 0.0 < params.tail < 0.5
                or params.trail_atr <= 0.0 or params.breakeven_pct <= 0.0
                or params.rejection_atr < 0.0):
            raise ValueError("Invalid strategy parameters")
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        side = 0
        entry = np.nan
        entry_bar = -1
        stop_profit = -np.inf
        armed = False
        for i in range(len(data)):
            if side:
                # A decision on the previous close fills at this bar's open.
                if np.isnan(entry):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        continue
                    entry = opens[i]
                    entry_bar = i
                    stop_profit = -params.trail_atr * atrs[i - 1]
                    armed = False
                if not np.isfinite(closes[i]) or not np.isfinite(atrs[i]) or atrs[i] <= 0:
                    side = 0
                    entry = np.nan
                    continue
                profit = side * (closes[i] - entry)
                # Stops are close-confirmed; the resulting exit fills next open.
                if profit <= stop_profit or i - entry_bar + 1 >= params.max_hold:
                    side = 0
                    entry = np.nan
                    continue
                if profit >= entry * params.breakeven_pct:
                    armed = True
                if armed:
                    # Profit-space stops only rise: short price stops only fall.
                    stop_profit = max(stop_profit, 0.0, profit - params.trail_atr * atrs[i])
            elif i + 1 < len(data) and np.isfinite(atrs[i]) and atrs[i] > 0:
                if longs[i] != shorts[i]:
                    side = 1 if longs[i] else -1
                    entry = np.nan
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(trail_atr=1.5)
    index = pd.date_range("2020-01-01", periods=22, freq="B")
    for side in (1, -1):
        closes = np.full(22, 100.0)
        closes[1] += side * 2.0
        data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0,
                             "close": closes, "volume": 1000.0}, index=index)
        indicators = pd.DataFrame({"atr": 1.0, "long_entry": False,
                                    "short_entry": False}, index=index)
        indicators.loc[index[0], "long_entry" if side == 1 else "short_entry"] = True
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
        assert result.data["signal"].iloc[:4].tolist() == [0, side, side, 0]
        data["close"] = 100.0
        result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
        assert result.data["signal"].iloc[1:19].eq(side).all()
        assert result.data["signal"].iloc[19:].eq(0).all()
    derived = GeneratedStrategy.indicators(None, data, Params())
    assert not derived["long_entry"].any() and not derived["short_entry"].any()


if __name__ == "__main__":
    _self_check()
