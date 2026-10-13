from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    profit_target: float = 0.025
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791197321"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 26

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"]
        previous = close.shift(1).where(close.shift(1) > 0)
        returns = (close / previous - 1).replace([np.inf, -np.inf], np.nan)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1) / previous
        recent_atr = true_range.rolling(5, min_periods=5).mean()
        prior_atr = true_range.shift(5).rolling(20, min_periods=20).mean()
        recent_std = returns.rolling(5, min_periods=5).std(ddof=0)
        prior_std = returns.shift(5).rolling(20, min_periods=20).std(ddof=0)
        width = (data["high"] - data["low"]).where(lambda x: x > 0)
        location = ((close - data["low"]) / width).clip(0, 1)
        out = pd.DataFrame(index=data.index)
        out["range_ratio"] = recent_atr / prior_atr.where(prior_atr > 0)
        out["noise_ratio"] = recent_std / prior_std.where(prior_std > 0)
        out["closing_strength"] = location.rolling(5, min_periods=5).mean()
        out["net_return"] = returns.rolling(5, min_periods=5).sum()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["range_ratio"] > 1.35)
            & (out["range_ratio"].shift(1) <= 1.35)
            & (out["noise_ratio"] <= 1.10)
            & (out["closing_strength"] > 0.65)
            & (out["net_return"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if params.max_hold < 1:
            raise ValueError("max_hold must be at least one")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        entry_price = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    entry_bar = i
            if active:
                target_hit = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold
                if target_hit or expired:
                    active = False
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [104.0] * 6,
        "low": [99.0] * 6,
        "close": [100.0, 100.0, 100.0, 100.0, 103.0, 103.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, True, False, False]}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, Params(max_hold=2))
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 0]
