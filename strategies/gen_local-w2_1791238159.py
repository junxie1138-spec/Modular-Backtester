from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    gap_tension: float = 0.25
    stop_atr: float = 1.75

    def __post_init__(self):
        if not np.isfinite(self.gap_tension) or self.gap_tension <= 0:
            raise ValueError("gap_tension must be finite and positive")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791238159"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return 21

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["prior_close"]).abs(),
             (data["low"] - out["prior_close"]).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["prior_atr"] = out["atr"].shift(1).where(lambda x: x > 0)
        out["corridor_high"] = data["high"].rolling(20, min_periods=20).max().shift(1)
        out["corridor_low"] = data["low"].rolling(20, min_periods=20).min().shift(1)
        out["center"] = (out["corridor_high"] + out["corridor_low"]) / 2
        out["width"] = (out["corridor_high"] - out["corridor_low"]).replace(0, np.nan)
        out["gap"] = data["open"] - out["prior_close"]
        out["tension"] = (
            out["gap"] * (data["open"] + out["prior_close"] - 2 * out["center"])
            / (out["prior_atr"] * out["width"])
        )
        out["open_position"] = 2 * (data["open"] - out["center"]) / out["width"]
        out["close_position"] = 2 * (data["close"] - out["center"]) / out["width"]
        stretched = out["tension"].ge(params.gap_tension)
        buy = stretched & out["gap"].lt(0) & out["open_position"].le(-0.85) & out["close_position"].le(-0.60)
        sell = stretched & out["gap"].gt(0) & out["open_position"].ge(0.85) & out["close_position"].ge(0.60)
        out["candidate"] = np.select([buy, sell], [1, -1], default=0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        raw = np.zeros(len(data), dtype=int)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction = 0
        entry_bar = -1
        stop = np.nan
        for i in range(len(data)):
            if direction:
                # The previous close's decision fills at this bar's open.
                if entry_bar == -1:
                    entry_bar = i
                    if not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        direction = 0
                    else:
                        stop = opens[i] - direction * params.stop_atr * atr[i - 1]
                if direction and (
                    not np.isfinite(closes[i])
                    or direction * (closes[i] - stop) < 0
                    or i - entry_bar + 1 >= 4
                ):
                    direction = 0
                    entry_bar = -1
            elif candidates[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                direction = int(candidates[i])
                entry_bar = -1
            raw[i] = direction
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=15)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.loc[index[2], "close"] = 98.0
    data.loc[index[6], "close"] = 102.0
    indicators = pd.DataFrame({"candidate": 0, "atr": 100.0}, index=index)
    indicators.loc[index[[0, 4, 8]], "candidate"] = [1, -1, 1]
    indicators.loc[index[[0, 4, 8]], "atr"] = 1.0
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, Params(stop_atr=1.5)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    actual = strategy.indicators(data, Params())
    assert actual["candidate"].eq(0).all()
    assert actual.index.equals(data.index)


if __name__ == "__main__":
    _self_check()
