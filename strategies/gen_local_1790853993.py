from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDebtParams:
    backlog_window: int = 3
    rank_window: int = 60
    entry_percentile: float = 0.85
    profit_target: float = 0.015
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapDebtParams]):
    strategy_id = "gen_local_1790853993"

    @classmethod
    def params_type(cls):
        return GapDebtParams

    def warmup_bars(self, params: GapDebtParams) -> int:
        return int(max(200, params.backlog_window + params.rank_window))

    def indicators(self, data: pd.DataFrame, params: GapDebtParams) -> pd.DataFrame:
        if params.backlog_window < 1 or params.rank_window < 2:
            raise ValueError("Lookbacks must be positive; rank_window must be at least two.")
        if not 0 < params.entry_percentile <= 1:
            raise ValueError("entry_percentile must be in (0, 1].")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap_down"] = ((out["previous_close"] - data["open"]) / out["previous_close"]).clip(lower=0)
        out["remaining_deficit"] = ((out["previous_close"] - data["close"]) / out["previous_close"]).clip(lower=0)
        out["unfilled_gap"] = np.minimum(out["gap_down"], out["remaining_deficit"])
        out["backlog"] = out["unfilled_gap"].rolling(params.backlog_window, min_periods=params.backlog_window).sum()
        out["backlog_rank"] = out["backlog"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["entry"] = (
            (out["backlog_rank"] >= params.entry_percentile)
            & (out["unfilled_gap"] > 0)
            & (data["close"] > data["open"])
            & (data["close"] > out["ma200"])
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapDebtParams) -> SignalFrame:
        if params.hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("hold_bars and profit_target must be positive.")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # The pending entry fills at this bar's open, never a future row.
                if i == entry_bar:
                    entry_price = opens[i]
                valid = np.isfinite(entry_price) and entry_price > 0
                # Targets are evaluated at the close; exits fill next bar.
                target_hit = valid and np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                timed_out = i - entry_bar + 1 >= params.hold_bars
                if not valid or target_hit or timed_out:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signals[i] = 1
            elif entries[i]:
                signals[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapDebtParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 103.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True] + [False] * 7}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    data.loc[index[1], "open"] = 103.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()
    assert not strategy.indicators(data, params)["entry"].any()
