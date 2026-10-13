from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRefractoryParams:
    contraction: float = 0.65
    stop_atr: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.contraction) or not 0 < self.contraction < 1:
            raise ValueError("contraction must be between zero and one")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be positive")


class GeneratedStrategy(BaseStrategy[GapRefractoryParams]):
    strategy_id = "gen_local-w2_1791237387"

    @classmethod
    def params_type(cls):
        return GapRefractoryParams

    @staticmethod
    def warmup_bars(params):
        return 22

    @staticmethod
    def indicators(data, params):
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["day_range"] = data["high"] - data["low"]
        out["range_baseline"] = out["day_range"].shift(1).rolling(20, min_periods=20).mean()
        out["true_range"] = pd.concat(
            [out["day_range"], (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["gap"] = data["open"] - previous_close
        out["prior_body"] = (data["close"] - data["open"]).shift(1)
        out["body"] = data["close"] - data["open"]
        prior_range = out["day_range"].shift(1)
        out["setup"] = (
            (prior_range > 1.8 * out["range_baseline"].shift(1))
            & (out["day_range"] < params.contraction * prior_range)
            & (out["gap"].abs() >= 0.15 * prior_range)
            & (out["gap"] * out["prior_body"] > 0)
            & (out["gap"] * out["body"] < 0)
            & (data["close"] >= data["low"].shift(1))
            & (data["close"] <= data["high"].shift(1))
            & (out["day_range"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        out["candidate"] = pd.Series(
            np.where(out["setup"], -np.sign(out["gap"].fillna(0)), 0),
            index=data.index
        ).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        candidates = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        position = pending = age = 0
        entry = distance = 0.0
        for i in range(len(data)):
            exited = False
            if pending:
                # Yesterday's decision fills at today's open; freeze yesterday's ATR.
                if np.isfinite(opens[i]) and opens[i] > 0:
                    position = pending
                    entry = opens[i]
                    distance = params.stop_atr * atrs[i - 1]
                    age = 0
                else:
                    exited = True
                pending = 0
            if position:
                age += 1
                stopped = (
                    not np.isfinite(closes[i])
                    or (position == 1 and closes[i] < entry - distance)
                    or (position == -1 and closes[i] > entry + distance)
                )
                if stopped or age >= 2:
                    position = 0
                    exited = True
                else:
                    raw[i] = position
            elif not exited:
                if (candidates[i] in (-1, 1)
                        and np.isfinite(atrs[i]) and atrs[i] > 0
                        and np.isfinite(closes[i]) and closes[i] > 0):
                    pending = int(candidates[i])
                    raw[i] = pending
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    params = GapRefractoryParams()
    ind = pd.DataFrame(
        {"atr": [2.0, 20.0, 2.0, 2.0, 2.0, 2.0],
         "candidate": [1, 0, 0, -1, 0, 0]}, index=index
    )
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    for side, adverse in ((1, 97.0), (-1, 103.0)):
        stopped_data = data.copy()
        stopped_data.iloc[1, stopped_data.columns.get_loc("close")] = adverse
        stopped_ind = ind.copy()
        stopped_ind["candidate"] = [side, 0, 0, 0, 0, 0]
        result = GeneratedStrategy.generate_signals(
            stopped_data, stopped_ind, None, params
        ).data
        assert result["signal"].tolist() == [0, side, 0, 0, 0, 0]
    for side in (1, -1):
        fixture = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0,
             "close": 100.0, "volume": 1000.0},
            index=pd.date_range("2020-01-01", periods=24)
        )
        fixture.iloc[21, fixture.columns.get_loc("high")] = 105.0
        fixture.iloc[21, fixture.columns.get_loc("low")] = 95.0
        fixture.iloc[21, fixture.columns.get_loc("close")] = 100.0 + side * 4
        fixture.iloc[22, fixture.columns.get_loc("open")] = 100.0 + side * 6
        fixture.iloc[22, fixture.columns.get_loc("close")] = 100.0 + side * 5
        fixture.iloc[22, fixture.columns.get_loc("high")] = 101.5 + side * 5.5
        fixture.iloc[22, fixture.columns.get_loc("low")] = 98.5 + side * 5.5
        computed = GeneratedStrategy.indicators(fixture, params)
        assert computed["candidate"].iloc[22] == -side
        result = GeneratedStrategy.generate_signals(
            fixture, computed, None, params
        ).data
        assert result["signal"].iloc[22] == 0
        assert result["signal"].iloc[23] == -side
    actual = GeneratedStrategy.indicators(data, params)
    assert actual["candidate"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) == 22


if __name__ == "__main__":
    _self_check()
