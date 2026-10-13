from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    impulse_floor: float = 0.65
    profit_target: float = 0.05


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791232631"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 84

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not np.isfinite(params.impulse_floor) or params.impulse_floor <= 0:
            raise ValueError("impulse_floor must be positive and finite")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        close = data["close"].astype(float)
        close = close.where(np.isfinite(close) & close.gt(0))
        weights = np.ones(64, dtype=float)
        for k in range(1, 64):
            weights[k] = -weights[k - 1] * (0.45 - k + 1) / k
        fractional = close * weights[0]
        for k in range(1, 64):
            fractional = fractional + close.shift(k) * weights[k]
        out = pd.DataFrame(index=data.index)
        out["fractional_close"] = fractional
        out["fractional_step"] = fractional.diff()
        out["noise"] = out["fractional_step"].abs().rolling(20, min_periods=20).mean()
        weekly = fractional.diff(5)
        block_names = []
        for k in range(4):
            name = f"weekly_{k}"
            out[name] = weekly.shift(5 * k)
            block_names.append(name)
        blocks = out[block_names]
        valid = blocks.notna().all(axis=1) & out["noise"].gt(0)
        magnitude = blocks.abs()
        out["balance"] = magnitude.min(axis=1) / magnitude.max(axis=1).replace(0, np.nan)
        out["strength"] = magnitude.min(axis=1) / (out["noise"] * np.sqrt(5.0))
        eligible = valid & out["balance"].ge(0.25) & out["strength"].ge(params.impulse_floor)
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[eligible & blocks.gt(0).all(axis=1)] = 1
        direction.loc[eligible & blocks.lt(0).all(axis=1)] = -1
        out["direction"] = direction
        out["entry_direction"] = direction.where(direction.ne(direction.shift(1).fillna(0)), 0).astype(int)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                valid_close = np.isfinite(closes[i]) and closes[i] > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry and valid_close else np.nan
                if not valid_entry or gain >= params.profit_target or i - entry_bar + 1 >= 18:
                    position = 0
                targets[i] = position
                continue
            if entries[i] in (-1, 1) and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=42)
    data = pd.DataFrame({"open": 100.0, "high": 107.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    data.loc[index[3], "close"] = 106.0
    indicators = pd.DataFrame({"entry_direction": 0}, index=index)
    indicators.loc[index[0], "entry_direction"] = 1
    indicators.loc[index[20], "entry_direction"] = -1
    result = GeneratedStrategy.generate_signals(data, indicators, None, Params()).data
    assert result["signal"].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    assert result["signal"].iloc[20] == 0
    assert result["signal"].iloc[21:39].eq(-1).all()
    assert result["signal"].iloc[39:].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
