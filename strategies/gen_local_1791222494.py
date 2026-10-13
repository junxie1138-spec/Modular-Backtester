from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 6
    efficiency_percentile: float = 0.85
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791222494"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if not 0.0 < params.efficiency_percentile <= 1.0:
            raise ValueError("efficiency_percentile must be in (0, 1]")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0.0:
            raise ValueError("profit_target must be positive and finite")
        if params.max_hold_bars not in (1, 2):
            raise ValueError("max_hold_bars must be 1 or 2")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["body"] = data["close"] - data["open"]
        out["travel_per_share"] = (
            out["body"].abs() / data["volume"].where(data["volume"] > 0.0)
        ).replace([np.inf, -np.inf], np.nan)
        out["efficiency_rank"] = out["travel_per_share"].rolling(
            params.lookback, min_periods=params.lookback
        ).rank(pct=True)
        out["volume_rank"] = data["volume"].where(
            data["volume"] > 0.0
        ).rolling(params.lookback, min_periods=params.lookback).rank(pct=True)
        out["upper"] = data["high"].rolling(
            params.lookback, min_periods=params.lookback
        ).max().shift(1)
        out["lower"] = data["low"].rolling(
            params.lookback, min_periods=params.lookback
        ).min().shift(1)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        ready = indicators.notna().all(axis=1)
        gate = (
            ready
            & indicators["efficiency_rank"].ge(params.efficiency_percentile)
            & indicators["volume_rank"].le(0.5)
        )
        long_entry = (
            gate & data["close"].gt(indicators["upper"])
            & indicators["body"].gt(0.0)
        )
        short_entry = (
            gate & data["close"].lt(indicators["lower"])
            & indicators["body"].lt(0.0)
        )
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        held = 0
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # A decision on the preceding close fills at this bar's open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                valid = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(closes[i]) and closes[i] > 0.0
                )
                gain = side * (closes[i] / entry_price - 1.0) if valid else np.nan
                # Targets are observed at the close; exits fill next bar.
                if not valid or gain >= params.profit_target or held >= params.max_hold_bars:
                    side = 0
                    held = 0
                    entry_price = np.nan
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0.0:
                side = int(entries[i])
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    for direction in (1, -1):
        close = np.array([100.0] * 6 + [
            100.0 + 2 * direction,
            100.0 + 3 * direction,
            100.0 + 3 * direction,
            100.0 + 3 * direction,
        ])
        opens = np.r_[100.0, close[:-1]]
        data = pd.DataFrame({
            "open": opens,
            "high": np.maximum(opens, close) + 0.25,
            "low": np.minimum(opens, close) - 0.25,
            "close": close,
            "volume": [10.0] * 6 + [1.0, 10.0, 10.0, 10.0],
        }, index=pd.date_range("2020-01-01", periods=10))
        for target in (0.005, 1.0):
            params = Params(profit_target=target)
            result = strategy.generate_signals(
                data, strategy.indicators(data, params), None, params
            ).data
            assert strategy.warmup_bars(params) <= 10
            assert result["signal"].iloc[:7].eq(0).all()
            assert result["signal"].iloc[7] == direction
            assert result["signal"].iloc[8] == (0 if target == 0.005 else direction)
            assert result["signal"].iloc[9] == 0
            assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
