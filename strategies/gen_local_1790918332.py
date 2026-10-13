from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    tension_z: float = 0.8
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790918332"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 95

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.tension_z) or params.tension_z <= 0:
            raise ValueError("tension_z must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        close = data["close"].astype(float)
        weights = np.ones(32)
        for lag in range(1, 32):
            weights[lag] = -weights[lag - 1] * (0.45 - lag + 1) / lag
        out = pd.DataFrame(index=data.index)
        fd = close * weights[0]
        for lag in range(1, 32):
            fd = fd + close.shift(lag) * weights[lag]
        out["fractional_close"] = fd
        out["anchor"] = fd.rolling(63, min_periods=63).mean().shift(1)
        out["dispersion"] = fd.rolling(63, min_periods=63).std().shift(1)
        out["tension"] = (fd - out["anchor"]) / out["dispersion"].replace(0, np.nan)
        out["release"] = fd.diff()
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["season"] = (
            (data.index.days_in_month - data.index.day <= 4)
            | (data.index.day <= 3)
        )
        out["enter"] = (
            out["season"]
            & (out["tension"] <= -params.tension_z)
            & (out["release"] > 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["enter"].fillna(False).to_numpy(dtype=bool)
        active = pending = armed = False
        entry = stop = entry_atr = 0.0
        age = 0
        for i in range(n):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    entry = opens[i]
                    stop = entry - 2.0 * entry_atr
                    age = 0
                    armed = False
                    active = True
            if active:
                age += 1
                valid = (
                    np.isfinite(closes[i]) and closes[i] > 0
                    and np.isfinite(lows[i])
                    and np.isfinite(atr[i]) and atr[i] > 0
                )
                # Stops observed today request an exit at tomorrow's open.
                # Test the existing stop before raising it at today's close.
                if not valid or lows[i] <= stop or age >= 4:
                    active = False
                    continue
                if closes[i] >= entry * 1.01:
                    armed = True
                if armed:
                    stop = max(stop, entry, closes[i] - params.trail_atr * atr[i])
                raw[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending = True
                entry_atr = atr[i]
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.bdate_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 99.5,
         "close": [100.0, 102.0, 101.0, 101.0, 101.0, 101.0, 101.0],
         "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"atr": 1.0, "enter": [True, False, False, False, False, False, False]},
        index=index
    )
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    data["close"] = 100.5
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[2], "low"] = 98.0
    result = strategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert (result["size"] > 0).all()
    history = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0},
        index=pd.bdate_range("2019-01-01", periods=120)
    )
    derived = strategy.indicators(history, Params())
    assert not derived["enter"].any()
    assert derived["anchor"].first_valid_index() == history.index[94]


if __name__ == "__main__":
    _self_check()
