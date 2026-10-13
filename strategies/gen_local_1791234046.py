from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GraphParams:
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GraphParams]):
    strategy_id = "gen_local_1791234046"

    @classmethod
    def params_type(cls):
        return GraphParams

    def warmup_bars(self, params: GraphParams) -> int:
        return 16

    def indicators(self, data: pd.DataFrame, params: GraphParams) -> pd.DataFrame:
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        out = pd.DataFrame(index=data.index)
        # Horizontal visibility: every interior close is below both endpoints.
        for span in range(2, 16):
            left = close.shift(span)
            interior = close.shift(1).rolling(span - 1, min_periods=span - 1).max()
            valid = left.notna() & close.notna() & interior.notna()
            visible = (interior.lt(left) & interior.lt(close)).where(valid)
            out[f"visible_{span}"] = visible.astype(float)
            out[f"up_{span}"] = (visible & close.gt(left)).where(valid).astype(float).rolling(16 - span, min_periods=16 - span).sum()
            out[f"down_{span}"] = (visible & close.lt(left)).where(valid).astype(float).rolling(16 - span, min_periods=16 - span).sum()
        out["up_edges"] = out[[f"up_{k}" for k in range(2, 16)]].sum(axis=1, min_count=14)
        out["down_edges"] = out[[f"down_{k}" for k in range(2, 16)]].sum(axis=1, min_count=14)
        out["edge_count"] = out["up_edges"] + out["down_edges"]
        out["bull_share"] = (out["up_edges"] / out["edge_count"].replace(0, np.nan)).fillna(0.0)
        out["window_return"] = (close / close.shift(15) - 1.0).fillna(0.0)
        out["ready"] = close.rolling(16, min_periods=16).count().eq(16)
        out["qualified"] = out["ready"] & out["edge_count"].ge(4) & out["bull_share"].ge(0.70) & out["window_return"].gt(0)
        out["entry"] = out["qualified"] & ~out["qualified"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GraphParams) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0 or params.max_hold < 1:
            raise ValueError("profit_target must be positive and max_hold at least one")
        decisions = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Capture the entry open only when its bar has actually arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        entry_bar = -1
                        continue
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                time_up = i - entry_bar + 1 >= params.max_hold
                if target_hit or time_up:
                    entry_bar = -1
                    continue
                decisions[i] = 1
            elif entries[i]:
                entry_bar = i + 1
                decisions[i] = 1
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0,
                         "close": [100.0, 100.0, 103.0, 100.0, 100.0, 100.0],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False]}, index=index)
    target = strategy.generate_signals(data, indicators, None, GraphParams()).data
    assert target["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    timed = strategy.generate_signals(data, indicators, None, GraphParams(profit_target=0.10, max_hold=3)).data
    assert timed["signal"].tolist() == [0, 1, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
