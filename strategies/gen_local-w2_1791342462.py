from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumMemoryParams:
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[MomentumMemoryParams]):
    strategy_id = "gen_local-w2_1791342462"

    @classmethod
    def params_type(cls) -> type[MomentumMemoryParams]:
        return MomentumMemoryParams

    def warmup_bars(self, params: MomentumMemoryParams) -> int:
        return 21

    def indicators(self, data: pd.DataFrame, params: MomentumMemoryParams) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        out = pd.DataFrame(index=data.index)
        out["return"] = close.pct_change(fill_method=None)
        out["roc"] = close.pct_change(3, fill_method=None)
        out["acceleration"] = out["roc"].diff(3)
        out["prior_acceleration"] = out["acceleration"].shift(1)
        out["prior_roc"] = out["roc"].shift(1)
        out["noise"] = out["return"].rolling(20, min_periods=20).std(ddof=0) * np.sqrt(6.0)
        out["prior_noise"] = out["noise"].shift(1)
        out["direction"] = np.sign(out["prior_acceleration"])
        out = out.replace([np.inf, -np.inf], np.nan)
        direction = out["direction"]
        burst = (out["prior_noise"] > 0) & (out["prior_acceleration"].abs() >= 1.5 * out["prior_noise"])
        cooling = (direction * out["acceleration"] >= 0) & (out["acceleration"].abs() <= 0.5 * out["prior_acceleration"].abs())
        retained = (direction * out["prior_roc"] > 0) & (direction * out["roc"] >= 0.8 * out["prior_roc"].abs())
        ready = out.notna().all(axis=1)
        out["entry"] = direction.where(ready & burst & cooling & retained, 0).fillna(0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumMemoryParams) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not isinstance(params.max_hold, int) or isinstance(params.max_hold, bool) or params.max_hold < 1:
            raise ValueError("max_hold must be a positive integer")
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            # Yesterday's decision fills at today's observed open.
            if position == 0 and i > 0 and decisions[i - 1] != 0:
                if np.isfinite(opens[i]) and opens[i] > 0:
                    position = int(decisions[i - 1])
                    entry_price = opens[i]
                    entry_bar = i
            if position != 0:
                gain = position * (closes[i] / entry_price - 1.0)
                expired = i - entry_bar + 1 >= params.max_hold
                if (np.isfinite(gain) and gain >= params.profit_target) or expired:
                    position = 0
                    decisions[i] = 0
                else:
                    decisions[i] = position
            elif np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = entries[i]
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=8)
    sample = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 97.0, "close": [100.0, 103.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": [1, 0, -1, 0, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    result = strategy.generate_signals(sample, entries, None, MomentumMemoryParams(profit_target=0.02, max_hold=3)).data
    assert result["signal"].tolist() == [0, 1, 0, -1, -1, -1, 0, 0]
    assert result.index.equals(index) and (result["size"] > 0).all()
    warmup = strategy.indicators(sample, MomentumMemoryParams())
    assert warmup["entry"].eq(0).all()
