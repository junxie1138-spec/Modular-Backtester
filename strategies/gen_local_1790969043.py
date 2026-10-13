from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    change_threshold: float = 2.5
    trail_atr: float = 0.75


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790969043"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.change_threshold) or params.change_threshold <= 0:
            raise ValueError("change_threshold must be positive and finite")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1)
        out["tr"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        out["baseline"] = out["tr"].rolling(20, min_periods=20).mean().shift(1)
        out["atr"] = out["tr"].rolling(5, min_periods=5).mean()
        out["surprise"] = (
            out["tr"] / out["baseline"].where(out["baseline"] > 0) - 1.15
        ).replace([np.inf, -np.inf], np.nan)
        out["cumulative"] = out["surprise"].fillna(0.0).cumsum()
        out["floor"] = out["cumulative"].cummin().clip(upper=0.0)
        out["cusum"] = out["cumulative"] - out["floor"]
        span = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["close_location"] = ((data["close"] - data["low"]) / span).fillna(0.5)
        out["entry"] = (
            out["surprise"].notna()
            & (out["atr"] > 0)
            & (out["cusum"] >= params.change_threshold)
            & (out["cusum"].shift(1).fillna(0.0) < params.change_threshold)
            & (data["close"] > previous)
            & (out["close_location"] >= 0.7)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry = peak = 0.0
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            valid = (
                np.isfinite([opens[i], highs[i], lows[i], closes[i], atrs[i]]).all()
                and min(opens[i], highs[i], lows[i], closes[i], atrs[i]) > 0
            )
            if not valid:
                active = False
                continue
            if not active and i > 0 and decisions[i - 1] == 1:
                # ponytail: modeled fill uses configured 5-bps slippage;
                # use actual fill callbacks if the framework exposes them.
                entry = opens[i] * 1.0005
                peak = entry
                stop = -np.inf
                age = 0
                armed = False
                active = True
            if active:
                age += 1
                # Only the stop established before this bar can inspect its low.
                hit = armed and lows[i] <= stop
                peak = max(peak, highs[i])
                if peak >= entry * 1.004:
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atrs[i])
                    # A freshly raised stop is checked against the close,
                    # avoiding an invented intraday high/low ordering.
                    hit = hit or closes[i] <= stop
                if hit or age >= 2:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i]:
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8,
         "close": 100.1, "volume": 1000.0}, index=index
    )
    ind = pd.DataFrame(
        {"atr": 1.0, "entry": [True, False, False, False]}, index=index
    )
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 1, 0]
    data.loc[index[1], "high"] = 102.0
    result = strategy.generate_signals(data, ind, None, Params()).data
    assert result["signal"].tolist() == [0, 1, 0, 0]
    assert result["size"].gt(0).all()
    assert not strategy.indicators(data, Params())["entry"].any()


if __name__ == "__main__":
    _self_check()
