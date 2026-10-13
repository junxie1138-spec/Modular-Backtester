from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    memory: int = 20
    profit_target: float = 0.012


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790933045"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.memory) + 6

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.memory < 3 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("memory must be >= 3 and profit_target must be positive")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        p = out["log_close"]
        # Five-close least-squares latent slope, using fixed centered weights.
        out["slope"] = (2 * p + p.shift(1) - p.shift(3) - 2 * p.shift(4)) / 10
        out["innovation"] = out["slope"].diff()
        out["lagged_innovation"] = out["innovation"].shift(1)
        out["persistence"] = out["innovation"].rolling(
            params.memory, min_periods=params.memory
        ).corr(out["lagged_innovation"]).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        armed = True
        active = False
        entry_bar = -1
        entry_price = np.nan
        values = indicators[["slope", "innovation", "persistence"]].to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        for i, (slope, innovation, persistence) in enumerate(values):
            valid = np.isfinite([slope, innovation, persistence]).all()
            if valid and persistence <= 0:
                armed = True
            if active:
                # The entry decision was made yesterday; today's open is now observable.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        active = False
                if active and (
                    closes[i] >= entry_price * (1 + params.profit_target)
                    or i - entry_bar + 1 >= 4
                ):
                    active = False
            elif valid and armed and persistence >= 0.2 and slope > 0 and innovation > 0:
                active = True
                armed = False
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = int(active)
        df["signal"] = signals
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=40)
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    forced = pd.DataFrame({"slope": 0.01, "innovation": 0.001, "persistence": 0.3}, index=index)
    time_exit = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert time_exit.iloc[:7].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[2], "close"] = 102.0
    target_exit = strategy.generate_signals(data, forced, ctx, params).data["signal"]
    assert target_exit.iloc[:5].tolist() == [0, 1, 1, 0, 0]
    flat = data.copy()
    flat["close"] = 100.0
    assert not strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].any()
    data["close"] = 100 * np.exp(np.cumsum(np.sin(np.arange(40) * 0.4) * 0.01))
    full = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(full.iloc[:30], strategy.indicators(data.iloc[:30], params))
    assert strategy.warmup_bars(params) >= params.memory + 6
