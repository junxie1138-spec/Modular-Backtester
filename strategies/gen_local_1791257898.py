from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapMemoryParams:
    frac_order: float = 0.55
    frac_window: int = 24
    rank_window: int = 126
    entry_quantile: float = 0.80
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GapMemoryParams]):
    strategy_id = "gen_local_1791257898"

    @classmethod
    def params_type(cls) -> type[GapMemoryParams]:
        return GapMemoryParams

    def warmup_bars(self, params: GapMemoryParams) -> int:
        return int(max(params.frac_window + params.rank_window + 1, params.rank_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapMemoryParams) -> pd.DataFrame:
        if not (0 < params.frac_order < 1 and params.frac_window >= 2
                and params.rank_window >= 2 and 0.5 < params.entry_quantile < 1
                and np.isfinite(params.profit_target) and params.profit_target > 0
                and params.max_hold >= 1):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        opening = data["open"].astype(float)
        log_close = np.log(close.where(close > 0)).replace([np.inf, -np.inf], np.nan)
        fractional_close = log_close.copy()
        weight = 1.0
        # ponytail: finite fractional memory; increase frac_window for longer tails.
        for k in range(1, params.frac_window):
            weight *= (k - 1 - params.frac_order) / k
            fractional_close += weight * log_close.shift(k)
        previous_fractional_close = fractional_close.shift(1)
        fractional_floor = previous_fractional_close.rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(1.0 - params.entry_quantile)
        previous_close = close.shift(1)
        gap = (opening / previous_close.where(previous_close > 0) - 1.0).replace(
            [np.inf, -np.inf], np.nan
        )
        gap_threshold = gap.shift(1).rolling(
            params.rank_window, min_periods=params.rank_window
        ).quantile(params.entry_quantile)
        entry = (
            (previous_fractional_close <= fractional_floor)
            & (fractional_close > previous_fractional_close)
            & (gap > gap_threshold)
            & (gap > 0)
            & (close >= opening)
            & (opening > 0)
        ).fillna(False)
        return pd.DataFrame({
            "log_close": log_close,
            "fractional_close": fractional_close,
            "previous_fractional_close": previous_fractional_close,
            "fractional_floor": fractional_floor,
            "previous_close": previous_close,
            "gap": gap,
            "gap_threshold": gap_threshold,
            "entry": entry,
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapMemoryParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active and entry_bar < 0:
                # The previous close's entry decision fills at this open.
                entry_price = opens[i]
                entry_bar = i
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                    signal[i] = 0
                    continue
            if active:
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                time_stop = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_stop:
                    active = False
            elif entries[i]:
                active = True
                entry_bar = -1
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GapMemoryParams(max_hold=3)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [80.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "high": [81.0, 104.0, 101.0, 101.0, 101.0, 101.0],
        "low": [79.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        "close": [80.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "entry": [True, False, False, False, False, False]
    }, index=index)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 103.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
