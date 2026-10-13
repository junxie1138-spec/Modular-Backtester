from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SurplusParams:
    fractional_order: float = 0.45
    fractional_window: int = 32
    baseline_window: int = 20
    budget_window: int = 5
    capacity: float = 3.0
    atr_window: int = 14
    stop_atr: float = 1.5
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[SurplusParams]):
    strategy_id = "gen_local_1791203760"

    @classmethod
    def params_type(cls) -> type[SurplusParams]:
        return SurplusParams

    @classmethod
    def warmup_bars(cls, params: SurplusParams) -> int:
        return int(max(
            params.fractional_window + params.baseline_window + params.budget_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: SurplusParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_window):
            weight *= (lag - 1 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["baseline"] = fractional.shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).median()
        out["pressure"] = ((fractional - out["baseline"]) / out["atr"].where(out["atr"] > 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        out["surplus"] = (out["pressure"] - 0.25).clip(lower=0.0)
        out["budget_load"] = out["surplus"].rolling(
            params.budget_window, min_periods=params.budget_window
        ).sum()
        out["overflow"] = (out["budget_load"] > params.capacity) & (out["pressure"] > 0)
        out["confirmed"] = out["overflow"] & out["overflow"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SurplusParams,
    ) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        active = False
        entry_bar = -1
        entry_atr = np.nan
        stop_price = np.nan
        for i in range(len(data)):
            if active:
                if entry_bar < 0:
                    # The previous close's decision fills at this bar's open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        active = False
                        continue
                    entry_bar = i
                    stop_price = opens[i] - params.stop_atr * entry_atr
                stopped = np.isfinite(closes[i]) and closes[i] < stop_price
                expired = i - entry_bar + 1 >= params.hold_days
                if stopped or expired:
                    active = False
                    entry_bar = -1
                else:
                    decisions[i] = 1
            elif confirmed[i] and np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                active = True
                entry_bar = -1
                entry_atr = atr[i]
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = SurplusParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": 1.0,
        "confirmed": [False, True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]
    data.loc[index[3], "close"] = 98.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    history = pd.DataFrame({
        "open": np.arange(100.0, 200.0),
        "high": np.arange(101.0, 201.0),
        "low": np.arange(99.0, 199.0),
        "close": np.arange(100.0, 200.0),
        "volume": 1000000.0,
    }, index=pd.date_range("2020-01-01", periods=100))
    derived = GeneratedStrategy.indicators(None, history, params)
    assert derived["confirmed"].equals(
        derived["overflow"] & derived["overflow"].shift(1, fill_value=False)
    )
    assert derived["confirmed"].any()
