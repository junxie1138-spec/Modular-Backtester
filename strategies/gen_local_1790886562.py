from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    feedback_window: int = 40
    feedback_threshold: float = 0.15
    drift_window: int = 10
    vol_window: int = 20
    target_vol: float = 0.12
    holding_bars: int = 18

    def __post_init__(self):
        for value in (self.feedback_window, self.drift_window, self.vol_window):
            if type(value) is not int or value < 2:
                raise ValueError("Lookbacks must be integers >= 2")
        if type(self.holding_bars) is not int or self.holding_bars < 1:
            raise ValueError("holding_bars must be a positive integer")
        if not np.isfinite(self.feedback_threshold) or not -1 < self.feedback_threshold < 1:
            raise ValueError("feedback_threshold must be finite and in (-1, 1)")
        if not np.isfinite(self.target_vol) or not 0 < self.target_vol <= 1:
            raise ValueError("target_vol must be finite and in (0, 1]")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790886562"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.feedback_window + 3, params.drift_window + 2, params.vol_window + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lagged_return"] = out["return"].shift(1)
        out["squared_return"] = out["return"].pow(2).replace([np.inf, -np.inf], np.nan)
        out["feedback"] = out["lagged_return"].rolling(
            params.feedback_window, min_periods=params.feedback_window
        ).corr(out["squared_return"]).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
        out["drift"] = out["return"].rolling(
            params.drift_window, min_periods=params.drift_window
        ).mean()
        out["annual_vol"] = out["return"].rolling(
            params.vol_window, min_periods=params.vol_window
        ).std(ddof=1) * np.sqrt(252.0)
        out["eligible"] = (
            (out["feedback"] > params.feedback_threshold)
            & (out["drift"] > 0)
            & np.isfinite(out["annual_vol"])
            & (out["annual_vol"] > 1e-8)
        )
        out["confirmed"] = out["eligible"] & out["eligible"].shift(1, fill_value=False)
        out["exposure"] = (
            params.target_vol / out["annual_vol"].where(out["annual_vol"] > 1e-8)
        ).replace([np.inf, -np.inf], np.nan).clip(0.01, 1.0).fillna(1.0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        exposure = indicators["exposure"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        size = np.ones(len(data), dtype=float)
        entry_bar = -1
        entry_size = 1.0
        for i in range(len(data)):
            if entry_bar >= 0:
                if i - entry_bar == params.holding_bars:
                    entry_bar = -1
                    continue
            elif confirmed[i]:
                entry_bar = i
                entry_size = float(exposure[i])
            if entry_bar >= 0:
                signal[i] = 1
                size[i] = entry_size
        df = pd.DataFrame({"signal": signal, "size": size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=params.holding_bars + 6, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    flat = strategy.indicators(data, params)
    assert not flat["confirmed"].any()
    assert strategy.generate_signals(data, flat, ctx, params).data["signal"].eq(0).all()
    forced = flat.copy()
    forced["confirmed"] = False
    forced.iloc[1, forced.columns.get_loc("confirmed")] = True
    forced["exposure"] = 0.25
    forced.iloc[2:, forced.columns.get_loc("exposure")] = 0.75
    result = strategy.generate_signals(data, forced, ctx, params).data
    active = np.flatnonzero(result["signal"].to_numpy())
    assert np.array_equal(active, np.arange(2, 2 + params.holding_bars))
    assert result["size"].iloc[active].eq(0.25).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
