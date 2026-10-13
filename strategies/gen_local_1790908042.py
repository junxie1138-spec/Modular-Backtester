from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ReleaseParams:
    rank_window: int = 20
    profit_target: float = 0.015


class GeneratedStrategy(BaseStrategy[ReleaseParams]):
    strategy_id = "gen_local_1790908042"

    @classmethod
    def params_type(cls):
        return ReleaseParams

    def warmup_bars(self, params: ReleaseParams) -> int:
        return int(params.rank_window) + 4

    def indicators(self, data: pd.DataFrame, params: ReleaseParams) -> pd.DataFrame:
        if params.rank_window < 5:
            raise ValueError("rank_window must be at least 5")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["width"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["relative_range"] = out["width"] / data["close"].where(data["close"] > 0)
        out["range_rank"] = out["relative_range"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["compressed"] = out["range_rank"].le(0.35).astype(float).where(out["range_rank"].notna())
        out["prior_compressions"] = out["compressed"].shift(1).rolling(4, min_periods=4).sum()
        out["ceiling"] = data["high"].shift(1).rolling(4, min_periods=4).max()
        out["floor"] = data["low"].shift(1).rolling(4, min_periods=4).min()
        out["close_location"] = (data["close"] - data["low"]) / out["width"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ReleaseParams) -> SignalFrame:
        ready = indicators["range_rank"].ge(0.85) & indicators["prior_compressions"].ge(3)
        long_entry = ready & data["close"].gt(indicators["ceiling"]) & indicators["close_location"].ge(0.75)
        short_entry = ready & data["close"].lt(indicators["floor"]) & indicators["close_location"].le(0.25)
        entries = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side != 0:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        side = 0
                gain = side * (closes[i] / entry_price - 1.0) if side != 0 else np.nan
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                time_expired = i - entry_bar + 1 >= 4
                if side == 0 or target_hit or time_expired:
                    side = 0
                    signals[i] = 0
                    continue
            elif entries[i] != 0:
                side = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _selfcheck() -> None:
    strategy = GeneratedStrategy()
    params = ReleaseParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"range_rank": 0.0, "prior_compressions": 3.0, "ceiling": 99.0, "floor": 99.0, "close_location": 1.0}, index=index)
    ind.loc[index[0], "range_rank"] = 1.0
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=24)
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 102.0
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    ind["ceiling"] = 101.0
    ind["floor"] = 101.0
    ind["close_location"] = 0.0
    data.loc[index[2], "close"] = 98.0
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0, 0]
    actual = strategy.indicators(data, params)
    assert strategy.generate_signals(data, actual, ctx, params).data["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _selfcheck()
