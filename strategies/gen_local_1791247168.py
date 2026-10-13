from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParams:
    streak_length: int = 2
    min_gap: float = 0.001
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapParams]):
    strategy_id = "gen_local_1791247168"

    @classmethod
    def params_type(cls) -> type[GapParams]:
        return GapParams

    def warmup_bars(self, params: GapParams) -> int:
        return int(max(200, params.streak_length + 1))

    def indicators(self, data: pd.DataFrame, params: GapParams) -> pd.DataFrame:
        if params.streak_length < 1 or not np.isfinite(params.min_gap) or params.min_gap <= 0:
            raise ValueError("streak_length must be positive and min_gap must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")

        out = pd.DataFrame(index=data.index)
        close = data["close"]
        opening = data["open"]
        out["previous_close"] = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["gap"] = (opening / out["previous_close"].where(out["previous_close"] > 0) - 1).replace([np.inf, -np.inf], np.nan)
        out["valid"] = (
            np.isfinite(opening) & opening.gt(0)
            & np.isfinite(close) & close.gt(0)
            & np.isfinite(out["previous_close"]) & out["previous_close"].gt(0)
            & np.isfinite(out["gap"])
        ).fillna(False)
        out["down_gap_rejected"] = (
            out["valid"] & out["gap"].le(-params.min_gap)
            & close.ge(out["previous_close"])
        ).fillna(False)
        out["up_gap_rejected"] = (
            out["valid"] & out["gap"].ge(params.min_gap)
            & close.le(out["previous_close"])
        ).fillna(False)
        for flag, count in (("down_gap_rejected", "down_streak"), ("up_gap_rejected", "up_streak")):
            active = out[flag]
            out[count] = active.astype(int).groupby((~active).cumsum()).cumsum().astype(int)

        ready = out["valid"] & np.isfinite(out["ma200"])
        out["entry"] = np.select(
            [
                ready & close.gt(out["ma200"]) & out["down_streak"].eq(params.streak_length),
                ready & close.lt(out["ma200"]) & out["up_streak"].eq(params.streak_length),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapParams) -> SignalFrame:
        if params.max_hold_bars not in (1, 2) or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Use a positive finite target and a one- or two-bar holding limit")

        openings = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held = 0

        for i in range(len(data)):
            if position:
                # A decision on the preceding close fills at this bar's open.
                if held == 0:
                    entry_price = openings[i]
                held += 1
                valid_mark = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1) if valid_mark else np.nan
                # Targets are confirmed at the close; flatten at the next open.
                if not valid_mark or gain >= params.profit_target or held >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
                    held = 0
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                entry_price = np.nan
                held = 0
            signals[i] = position

        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapParams()
    index = pd.date_range("2020-01-01", periods=210, freq="B")
    close = np.full(210, 100.0)
    close[:199] = 99.0
    opening = close.copy()
    opening[200:202] = 99.8
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, close) + 1, "low": np.minimum(opening, close) - 1, "close": close, "volume": 1000000.0}, index=index)
    timed = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert timed["signal"].iloc[:202].eq(0).all()
    assert timed["signal"].iloc[202:205].tolist() == [1, 1, 0]

    data.loc[index[202], "close"] = 100.7
    targeted = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert targeted["signal"].iloc[202:204].tolist() == [1, 0]

    bearish = data.copy()
    bearish.loc[:, "close"] = 100.0
    bearish.loc[index[199]:, "close"] = 98.0
    bearish.loc[:, "open"] = bearish["close"]
    bearish.loc[index[200:202], "open"] = 98.2
    bearish.loc[:, "high"] = bearish[["open", "close"]].max(axis=1) + 1
    bearish.loc[:, "low"] = bearish[["open", "close"]].min(axis=1) - 1
    short = strategy.generate_signals(bearish, strategy.indicators(bearish, params), None, params).data
    assert short["signal"].iloc[202:205].tolist() == [-1, -1, 0]
    assert targeted["signal"].isin([-1, 0, 1]).all()
    assert targeted["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
