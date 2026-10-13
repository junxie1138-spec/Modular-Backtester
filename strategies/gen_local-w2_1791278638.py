from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalTransitionParams:
    lookback: int = 12
    min_strength: float = 0.50
    profit_target: float = 0.01
    hold_bars: int = 2
    min_size: float = 0.20
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[OrdinalTransitionParams]):
    strategy_id = "gen_local-w2_1791278638"

    @classmethod
    def params_type(cls) -> type[OrdinalTransitionParams]:
        return OrdinalTransitionParams

    @staticmethod
    def warmup_bars(params: OrdinalTransitionParams) -> int:
        return int(params.lookback) + 3

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalTransitionParams) -> pd.DataFrame:
        if params.lookback < 2 or not 0 < params.min_strength < 1:
            raise ValueError("lookback >= 2 and 0 < min_strength < 1 required")
        out = pd.DataFrame(index=data.index)
        midpoint = ((data["high"] + data["low"]) / 2).where(
            (data["high"] >= data["low"]) & (data["low"] > 0)
        )
        out["midpoint"] = midpoint.replace([np.inf, -np.inf], np.nan)
        a, b, c = out["midpoint"].shift(2), out["midpoint"].shift(1), out["midpoint"]
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        out["pattern"] = (
            4 * a.lt(b).astype(int) + 2 * a.lt(c).astype(int) + b.lt(c).astype(int)
        ).where(valid)
        out["previous_pattern"] = out["pattern"].shift(1)
        pairs_valid = out["pattern"].notna() & out["previous_pattern"].notna()
        out["pair_count"] = pairs_valid.astype(float).rolling(
            params.lookback, min_periods=params.lookback
        ).sum()
        joint_collision = pd.Series(0.0, index=data.index)
        origin_collision = pd.Series(0.0, index=data.index)
        codes = (0, 1, 3, 4, 6, 7)
        for origin in codes:
            origin_mass = pd.Series(0.0, index=data.index)
            for destination in codes:
                count = (
                    out["previous_pattern"].eq(origin) & out["pattern"].eq(destination)
                ).astype(float).rolling(params.lookback, min_periods=params.lookback).sum()
                out[f"count_{origin}_{destination}"] = count
                joint_collision = joint_collision + count.pow(2)
                origin_mass = origin_mass + count
            out[f"origin_count_{origin}"] = origin_mass
            origin_collision = origin_collision + origin_mass.pow(2)
        out["joint_collision"] = joint_collision
        out["origin_collision"] = origin_collision
        out["predictability"] = (
            joint_collision / origin_collision.replace(0, np.nan)
        ).where(out["pair_count"].eq(params.lookback))
        out["midpoint_step"] = out["midpoint"].diff()
        out["path_length"] = out["midpoint_step"].abs().rolling(
            params.lookback, min_periods=params.lookback
        ).sum()
        out["displacement"] = out["midpoint"] - out["midpoint"].shift(params.lookback)
        out["efficiency"] = (
            out["displacement"].abs() / out["path_length"].replace(0, np.nan)
        ).clip(0, 1)
        out["direction"] = np.sign(out["displacement"]).fillna(0).astype(int)
        out["strength"] = (out["predictability"] * out["efficiency"]).clip(0, 1).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalTransitionParams,
    ) -> SignalFrame:
        if not 1 <= params.hold_bars <= 2 or params.profit_target <= 0:
            raise ValueError("hold_bars must be 1 or 2; profit_target must be positive")
        if not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("require 0 < min_size <= max_size <= 1")
        strength = indicators["strength"].fillna(0).to_numpy()
        direction = indicators["direction"].fillna(0).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position, entry_bar, entry_price, position_size = 0, -1, np.nan, params.min_size
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    position = 0
                else:
                    gain = position * (closes[i] / entry_price - 1.0)
                    # Targets are close-confirmed; the exit executes at the next open.
                    if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                        position = 0
            elif (
                np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(strength[i]) and strength[i] >= params.min_strength
                and direction[i] in (-1, 1)
            ):
                position = int(direction[i])
                entry_bar = i + 1
                entry_price = np.nan
                position_size = params.min_size + (
                    params.max_size - params.min_size
                ) * float(np.clip(strength[i], 0, 1))
            signals[i] = position
            if position:
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100, 100, 100, 102, 102, 102, 102],
        "high": [101, 101, 103, 103, 103, 103, 103],
        "low": [99, 99, 99, 101, 101, 101, 101],
        "close": [100, 100, 102, 102, 102, 102, 102],
        "volume": 1000,
    }, index=index)
    params = OrdinalTransitionParams()
    forced = pd.DataFrame({"strength": 1.0, "direction": 1}, index=index)
    result = GeneratedStrategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    params.profit_target = 1.0
    params.hold_bars = 1
    result = GeneratedStrategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 0, 1, 0]
    forced["direction"] = -1
    data["close"] = [100, 100, 98, 98, 98, 98, 98]
    data["open"] = [100, 100, 100, 98, 98, 98, 98]
    params.profit_target = 0.01
    params.hold_bars = 2
    result = GeneratedStrategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, -1, -1, 0]
    warm = GeneratedStrategy.indicators(data, params)
    assert warm["strength"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
