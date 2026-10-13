from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownSeasonalityParams:
    drawdown_window: int = 20
    baseline_window: int = 63
    depth_excess: float = 0.008
    profit_target: float = 0.006
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[DrawdownSeasonalityParams]):
    strategy_id = "gen_local_1791319796"

    @classmethod
    def params_type(cls):
        return DrawdownSeasonalityParams

    def warmup_bars(self, params: DrawdownSeasonalityParams) -> int:
        return max(200, params.drawdown_window + params.baseline_window)

    def indicators(self, data: pd.DataFrame, params: DrawdownSeasonalityParams) -> pd.DataFrame:
        if params.drawdown_window < 1 or params.baseline_window < 1:
            raise ValueError("Lookbacks must be positive.")
        if not 1 <= params.max_hold_bars <= 2:
            raise ValueError("max_hold_bars must be 1 or 2.")
        if not np.isfinite(params.depth_excess) or params.depth_excess <= 0:
            raise ValueError("depth_excess must be finite and positive.")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive.")
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        out = pd.DataFrame(index=data.index)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["peak"] = close.rolling(params.drawdown_window, min_periods=params.drawdown_window).max()
        out["drawdown"] = 1.0 - close / out["peak"]
        out["baseline_depth"] = out["drawdown"].shift(1).rolling(
            params.baseline_window, min_periods=params.baseline_window
        ).mean()
        out["depth_overload"] = out["drawdown"] - out["baseline_depth"]
        out["weekday"] = data.index.dayofweek
        ready = out[["ma200", "drawdown", "baseline_depth"]].notna().all(axis=1)
        overload = ready & out["depth_overload"].ge(params.depth_excess)
        long_entry = overload & out["weekday"].eq(1) & close.gt(out["ma200"])
        short_entry = overload & out["weekday"].eq(3) & close.lt(out["ma200"])
        out["entry"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DrawdownSeasonalityParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # The preceding close's decision fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        side = 0
                        continue
                held_bars = i - entry_bar + 1
                gain = side * (closes[i] / entry_price - 1.0)
                if (np.isfinite(gain) and gain >= params.profit_target) or held_bars >= params.max_hold_bars:
                    side = 0
            elif entries[i]:
                side = int(entries[i])
                entry_bar = i + 1
            decisions[i] = side
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2024-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0,
         "close": [100.0, 100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0],
         "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame({"entry": [0, 1, 0, -1, 0, 0, 0, 0]}, index=index)
    strategy = GeneratedStrategy()
    params = DrawdownSeasonalityParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, -1, -1, 0, 0]
    data.loc[index[4], "close"] = 99.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, -1, 0, 0, 0]
    derived = strategy.indicators(data, params)
    assert derived["entry"].eq(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
