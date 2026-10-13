from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    percentile_window: int = 60
    exhaustion_quantile: float = 0.85
    profit_target: float = 0.006
    max_hold: int = 2
    size: float = 0.95

    def __post_init__(self):
        if self.percentile_window < 2 or self.max_hold not in (1, 2):
            raise ValueError("Require percentile_window >= 2 and max_hold in (1, 2)")
        if not 0.0 < self.exhaustion_quantile < 1.0:
            raise ValueError("exhaustion_quantile must be between zero and one")
        if not np.isfinite(self.profit_target) or self.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not np.isfinite(self.size) or self.size <= 0:
            raise ValueError("size must be finite and positive")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791341143"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(params.percentile_window + 3)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["range"] = (data["high"] - data["low"]).where(lambda s: s > 0)
        out["new_low_depth"] = (data["low"].shift(1) - data["low"]).clip(lower=0)
        # Wide auctions with little fresh downside imply exhausted propagation.
        out["recycled_range"] = (
            (out["range"] - out["new_low_depth"]).clip(lower=0)
            / out["previous_close"]
        )
        out["exhaustion_threshold"] = (
            out["recycled_range"].shift(1)
            .rolling(params.percentile_window, min_periods=params.percentile_window)
            .quantile(params.exhaustion_quantile)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["entry"] = (
            ready
            & (out["recycled_range"] > out["exhaustion_threshold"])
            & (out["range"] > out["range"].shift(1))
            & (out["new_low_depth"] > 0)
            & (out["new_low_depth"] < out["new_low_depth"].shift(1))
            & (data["close"] < out["previous_close"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Observe the entry open only when its bar has actually arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                invalid_fill = not np.isfinite(entry_price) or entry_price <= 0
                target_hit = (
                    not invalid_fill
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                timed_out = i - entry_bar + 1 >= params.max_hold
                if invalid_fill or target_hit or timed_out:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    raw[i] = 1
            elif entries[i]:
                entry_bar = i + 1
                raw[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = float(params.size)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0},
        index=index,
    )
    entry = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    signals = strategy.generate_signals(data, entry, ctx, params).data["signal"]
    assert signals.tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 100.7
    signals = strategy.generate_signals(data, entry, ctx, params).data["signal"]
    assert signals.tolist() == [0, 1, 0, 0, 0]
    assert not strategy.indicators(data, params)["entry"].any()
