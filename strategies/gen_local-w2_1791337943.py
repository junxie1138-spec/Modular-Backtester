from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAccelerationParams:
    lookback: int = 63
    acceleration_quantile: float = 0.85
    profit_target: float = 0.008
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapAccelerationParams]):
    strategy_id = "gen_local-w2_1791337943"

    @classmethod
    def params_type(cls) -> type[GapAccelerationParams]:
        return GapAccelerationParams

    def warmup_bars(self, params: GapAccelerationParams) -> int:
        if params.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0.5 < params.acceleration_quantile < 1.0:
            raise ValueError("acceleration_quantile must be between 0.5 and 1")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        return int(params.lookback + 3)

    def indicators(self, data: pd.DataFrame, params: GapAccelerationParams) -> pd.DataFrame:
        self.warmup_bars(params)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        gap = data["open"] / previous_close - 1.0
        acceleration = gap.diff()
        acceleration_magnitude = acceleration.abs()
        threshold = acceleration_magnitude.shift(1).rolling(
            params.lookback, min_periods=params.lookback
        ).quantile(params.acceleration_quantile)
        displacement = data["open"] - previous_close
        retained_gap = (data["close"] - previous_close) / displacement.replace(0.0, np.nan)
        return pd.DataFrame({
            "gap": gap,
            "gap_acceleration": acceleration,
            "acceleration_magnitude": acceleration_magnitude,
            "acceleration_threshold": threshold,
            "retained_gap": retained_gap,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapAccelerationParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        ready = indicators.notna().all(axis=1)
        expanding = indicators["gap"] * indicators["gap_acceleration"] > 0.0
        extreme = indicators["acceleration_magnitude"] > indicators["acceleration_threshold"]
        absorbed = (indicators["retained_gap"] > 0.0) & (indicators["retained_gap"] < 1.0)
        eligible = ready & expanding & extreme & absorbed
        entries = np.where(eligible, -np.sign(indicators["gap"].fillna(0.0)), 0).astype(int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan

        for i in range(len(data)):
            if position:
                # The previous close's decision fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0.0
                profit = position * (closes[i] / entry_price - 1.0) if valid_price else np.nan
                target_hit = np.isfinite(profit) and profit >= params.profit_target
                time_up = i - entry_bar + 1 >= params.hold_bars
                if target_hit or time_up or not valid_price:
                    position = 0
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0 and i + 1 < len(data):
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = object.__new__(GeneratedStrategy)
    params = GapAccelerationParams(lookback=2)
    index = pd.date_range("2020-01-01", periods=5)
    for side in (-1, 1):
        for gain in (0.0, 0.02):
            data = pd.DataFrame({
                "open": 100.0,
                "high": 103.0,
                "low": 97.0,
                "close": 100.0 * (1.0 + side * gain),
                "volume": 1000.0,
            }, index=index)
            indicators = pd.DataFrame({
                "gap": -side * 0.02,
                "gap_acceleration": -side * 0.01,
                "acceleration_magnitude": 0.01,
                "acceleration_threshold": 0.005,
                "retained_gap": 0.5,
            }, index=index)
            result = strategy.generate_signals(data, indicators, None, params).data
            expected = [0, side, 0] if gain else [0, side, side, 0]
            assert result["signal"].iloc[:len(expected)].tolist() == expected
            assert (result["size"] > 0.0).all()
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0.0}, index=index)
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert result["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
