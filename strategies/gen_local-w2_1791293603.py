from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_length: int = 3
    contraction_ratio: float = 0.9
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791293603"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return max(200, params.streak_length + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.streak_length < 1 or not 0 < params.contraction_ratio <= 1:
            raise ValueError("Invalid streak length or contraction ratio")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Profit target must be positive and finite")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("Holding period must be one or two bars")
        out = pd.DataFrame(index=data.index)
        prices = data[["open", "high", "low", "close"]]
        out["valid"] = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1) & (data["high"] >= data["low"])
        out["ma200"] = data["close"].where(out["valid"]).rolling(200, min_periods=200).mean()
        out["daily_range"] = (data["high"] - data["low"]).where(out["valid"])
        out["midpoint"] = ((data["high"] + data["low"]) / 2).where(out["valid"])
        out["lower_half"] = out["valid"] & (data["close"] < out["midpoint"]) & (out["daily_range"] > 0)
        out["streak"] = out["lower_half"].astype(int).groupby((~out["lower_half"]).cumsum()).cumsum()
        out["range_ratio"] = out["daily_range"] / out["daily_range"].shift(1).replace(0, np.nan)
        ready = out["valid"] & out["ma200"].notna() & np.isfinite(out["range_ratio"])
        trigger = out["streak"].eq(params.streak_length)
        out["entry"] = np.select(
            [
                ready & trigger & (data["close"] > out["ma200"]) & (out["range_ratio"] <= params.contraction_ratio),
                ready & trigger & (data["close"] < out["ma200"]) & (out["range_ratio"] >= 1 / params.contraction_ratio),
            ],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry_price = np.nan
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        valid = indicators["valid"].to_numpy(dtype=bool)
        for i in range(len(data)):
            if side:
                # Yesterday's entry decision fills at today's open.
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side, age, entry_price = 0, 0, np.nan
                        continue
                    entry_price = opens[i]
                age += 1
                gain = side * (closes[i] / entry_price - 1)
                # Close decisions execute at the following open, like entries.
                if not valid[i] or gain >= params.profit_target or age >= params.max_hold_bars:
                    side, age, entry_price = 0, 0, np.nan
                signals[i] = side
            elif valid[i] and entries[i]:
                side, age, entry_price = int(entries[i]), 0, np.nan
                signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    idx = pd.date_range("2020-01-01", periods=206)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=idx)
    data.loc[idx[199], ["open", "high", "low", "close"]] = [102.0, 105.0, 99.0, 101.0]
    data.loc[idx[200], ["open", "high", "low", "close"]] = [102.0, 104.0, 100.0, 101.0]
    data.loc[idx[201], ["open", "close"]] = [101.0, 102.0]
    strategy, params = GeneratedStrategy(), Params(streak_length=2)
    result = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert result["signal"].iloc[:201].eq(0).all()
    assert result["signal"].iloc[201:203].tolist() == [1, 0]
    data.loc[idx[201:203], "close"] = 101.0
    result = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert result["signal"].iloc[201:204].tolist() == [1, 1, 0]
    data.loc[idx[199], ["open", "high", "low", "close"]] = [98.0, 100.0, 96.0, 97.0]
    data.loc[idx[200], ["open", "high", "low", "close"]] = [98.0, 101.0, 95.0, 97.0]
    data.loc[idx[201], ["open", "close"]] = [97.0, 96.0]
    result = strategy.generate_signals(data, strategy.indicators(data, params), None, params).data
    assert result["signal"].iloc[201:203].tolist() == [-1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
