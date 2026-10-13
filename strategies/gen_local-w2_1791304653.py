from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.45
    fractional_window: int = 32
    correlation_window: int = 12
    memory_window: int = 6
    negative_level: float = -0.25
    switch_level: float = 0.10
    volatility_window: int = 20
    target_volatility: float = 0.10
    profit_target: float = 0.012
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791304653"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(200, params.fractional_window + params.correlation_window
                       + params.memory_window + 3, params.volatility_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0
                and params.fractional_window >= 2
                and params.correlation_window >= 3
                and params.memory_window >= 1
                and params.volatility_window >= 2
                and -1.0 < params.negative_level < 0.0
                and 0.0 <= params.switch_level < 1.0
                and 3 <= params.max_hold <= 5
                and np.isfinite(params.target_volatility)
                and params.target_volatility > 0.0
                and np.isfinite(params.profit_target)
                and params.profit_target > 0.0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["clean_close"] = close.where(np.isfinite(close) & close.gt(0.0))
        close = out["clean_close"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        fractional = close.copy()
        weight = 1.0
        # Fixed-width causal fractional difference of the close.
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["fractional_impulse"] = fractional.diff() / close.shift(1)
        out["lagged_impulse"] = out["fractional_impulse"].shift(1)
        out["serial_correlation"] = out["fractional_impulse"].rolling(
            params.correlation_window, min_periods=params.correlation_window
        ).corr(out["lagged_impulse"]).replace([np.inf, -np.inf], np.nan)
        out["recent_negative_correlation"] = out["serial_correlation"].rolling(
            params.memory_window, min_periods=params.memory_window
        ).min().shift(1)
        out["returns"] = close.pct_change(fill_method=None)
        out["annual_volatility"] = out["returns"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        out["exposure"] = (params.target_volatility / out["annual_volatility"].clip(
            lower=1e-6
        )).clip(lower=0.05, upper=0.95).fillna(0.05)
        out["regime"] = close.gt(out["ma200"])
        corr = out["serial_correlation"]
        out["entry"] = (
            out["regime"]
            & out["fractional_impulse"].gt(0.0)
            & corr.gt(params.switch_level)
            & corr.shift(1).le(params.switch_level)
            & out["recent_negative_correlation"].lt(params.negative_level)
            & out["annual_volatility"].gt(0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        n = len(data)
        decisions = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.05, dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        exposures = indicators["exposure"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        held_size = 0.05
        warmup = GeneratedStrategy.warmup_bars(params)
        for i in range(n):
            if active:
                # The entry decision was made yesterday; today's open is now known.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                target_hit = (
                    np.isfinite(closes[i]) and np.isfinite(entry_price)
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_up = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_up:
                    active = False
            elif i >= warmup - 1 and entries[i]:
                active = True
                entry_bar = i + 1
                entry_price = np.nan
                held_size = float(exposures[i]) if np.isfinite(exposures[i]) else 0.05
                held_size = min(0.95, max(0.05, held_size))
            decisions[i] = int(active)
            sizes[i] = held_size if active else 0.05
        df = pd.DataFrame({"signal": decisions, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params()
    index = pd.date_range("2020-01-01", periods=220)
    data = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    ind = pd.DataFrame({"entry": False, "exposure": 0.5}, index=index)
    ind.loc[index[200], "entry"] = True
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[200:206].tolist() == [0, 1, 1, 1, 1, 0]
    assert result["size"].iloc[201:205].eq(0.5).all()
    data.loc[index[202], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[200:205].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[201], "open"] = 102.0
    data.loc[index[202], "close"] = 103.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[201:206].tolist() == [1, 1, 1, 1, 0]
    flat = data.copy()
    flat["open"] = flat["high"] = flat["low"] = flat["close"] = 100.0
    actual = GeneratedStrategy.indicators(flat, params)
    result = GeneratedStrategy.generate_signals(flat, actual, None, params).data
    assert result["signal"].eq(0).all()
    assert np.isfinite(result["size"]).all() and result["size"].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
