from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangePressureParams:
    lookback: int = 16
    coupling_threshold: float = 0.35
    efficiency_floor: float = 0.25
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[RangePressureParams]):
    strategy_id = "gen_local_1791330988"

    @classmethod
    def params_type(cls) -> type[RangePressureParams]:
        return RangePressureParams

    @staticmethod
    def warmup_bars(params: RangePressureParams) -> int:
        return max(2, int(params.lookback)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangePressureParams) -> pd.DataFrame:
        window = max(2, int(params.lookback))
        result = pd.DataFrame(index=data.index)
        close = data["close"]
        result["range_width"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"]
        )
        result["relative_range"] = result["range_width"] / close.shift(1).where(
            close.shift(1) > 0
        )
        result["closing_pressure"] = (
            (2.0 * close - data["high"] - data["low"]) / result["range_width"]
        ).clip(-1.0, 1.0)
        result["range_pressure_coupling"] = result["relative_range"].rolling(
            window, min_periods=window
        ).corr(result["closing_pressure"]).replace([np.inf, -np.inf], np.nan)
        result["close_change"] = close.diff()
        result["path_length"] = result["close_change"].abs().rolling(
            window, min_periods=window
        ).sum()
        result["displacement"] = close - close.shift(window)
        result["efficiency"] = (
            result["displacement"].abs() / result["path_length"].where(
                result["path_length"] > 0
            )
        ).clip(0.0, 1.0)
        result["ready"] = (
            result["range_pressure_coupling"].notna()
            & result["efficiency"].notna()
            & close.notna()
            & (close > 0)
        )
        coupling = result["range_pressure_coupling"].fillna(0.0).clip(-1.0, 1.0)
        efficiency = result["efficiency"].fillna(0.0)
        direction = np.sign(result["displacement"].fillna(0.0))
        result["entry_direction"] = direction.where(
            result["ready"]
            & (coupling.abs() >= params.coupling_threshold)
            & (efficiency >= params.efficiency_floor)
            & (coupling * direction > 0),
            0,
        ).astype(int)
        result["conviction"] = (coupling.abs() * efficiency / 0.5).clip(0.2, 1.0)
        return result

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangePressureParams,
    ) -> SignalFrame:
        if params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("profit_target and max_hold_bars must be positive")
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].to_numpy(dtype=int)
        conviction = indicators["conviction"].to_numpy(dtype=float)
        position = 0
        entry_price = np.nan
        fill_bar = -1
        position_size = 1.0
        for i in range(len(data)):
            if position != 0:
                if np.isnan(entry_price):
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        continue
                    entry_price = opens[i]
                    fill_bar = i
                gain = position * (closes[i] / entry_price - 1.0)
                age = i - fill_bar + 1
                if gain >= params.profit_target or age >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
                    continue
            elif entries[i] != 0:
                position = int(entries[i])
                entry_price = np.nan
                position_size = float(conviction[i])
            signals[i] = position
            if position != 0:
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")
