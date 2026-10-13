from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MemoryPositionParams:
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[MemoryPositionParams]):
    strategy_id = "gen_local_1790861337"

    @classmethod
    def params_type(cls):
        return MemoryPositionParams

    def warmup_bars(self, params: MemoryPositionParams) -> int:
        return 109

    def indicators(self, data: pd.DataFrame, params: MemoryPositionParams) -> pd.DataFrame:
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        fractional_close = close.copy()
        weight = 1.0
        # Fixed 64-tap approximation of (1 - lag)**0.4.
        # ponytail: finite memory; increase taps if truncation sensitivity matters.
        for lag in range(1, 64):
            weight *= (lag - 1 - 0.4) / lag
            fractional_close = fractional_close + weight * close.shift(lag)

        fractional_floor = fractional_close.rolling(42, min_periods=42).min().shift(1)
        fractional_ceiling = fractional_close.rolling(42, min_periods=42).max().shift(1)
        price_floor = close.rolling(42, min_periods=42).min().shift(1)
        price_ceiling = close.rolling(42, min_periods=42).max().shift(1)
        fractional_position = (
            (fractional_close - fractional_floor)
            / (fractional_ceiling - fractional_floor).replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        price_position = (
            (close - price_floor)
            / (price_ceiling - price_floor).replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        position_gap = fractional_position - price_position
        leadership = (
            fractional_position.ge(0.80)
            & price_position.between(0.30, 0.70)
            & position_gap.ge(0.25)
        )
        entry_ready = leadership.rolling(3, min_periods=3).sum().eq(3)
        return pd.DataFrame(
            {
                "clean_close": close,
                "fractional_close": fractional_close,
                "fractional_floor": fractional_floor,
                "fractional_ceiling": fractional_ceiling,
                "price_floor": price_floor,
                "price_ceiling": price_ceiling,
                "fractional_position": fractional_position,
                "price_position": price_position,
                "position_gap": position_gap,
                "leadership": leadership,
                "entry_ready": entry_ready,
            },
            index=data.index,
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MemoryPositionParams,
    ) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not isinstance(params.max_hold_bars, int) or isinstance(params.max_hold_bars, bool) or params.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be a positive integer")

        decisions = np.zeros(len(data), dtype=int)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # Observe the entry open only once its bar has arrived.
                    entry_price = opens[i]
                expired = i - entry_bar + 1 >= params.max_hold_bars
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if not np.isfinite(entry_price) or entry_price <= 0 or expired or target_hit:
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif ready[i] and np.isfinite(closes[i]) and closes[i] > 0:
                entry_bar = i + 1
                decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = MemoryPositionParams()
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry_ready": False}, index=index)
    indicators.iloc[0, 0] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=109)
    signal = strategy.generate_signals(data, indicators, ctx, params).data["signal"]
    assert signal.iloc[0] == 0 and signal.iloc[1:19].eq(1).all()
    assert signal.iloc[19:].eq(0).all()
    data.loc[index[2], "close"] = 105.0
    signal = strategy.generate_signals(data, indicators, ctx, params).data["signal"]
    assert signal.iloc[1:3].eq(1).all() and signal.iloc[3:].eq(0).all()
    assert not strategy.indicators(data, params)["entry_ready"].any()


if __name__ == "__main__":
    _self_check()
