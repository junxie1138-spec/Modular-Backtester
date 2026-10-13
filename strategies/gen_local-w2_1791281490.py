from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    peak_window: int = 10
    rank_window: int = 40
    tail_percentile: float = 0.60
    depth_percentile: float = 0.55


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local-w2_1791281490"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        if params.peak_window < 2 or params.rank_window < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.5 < params.tail_percentile < 1.0:
            raise ValueError("tail_percentile must be between 0.5 and 1")
        if not 0.0 < params.depth_percentile < 1.0:
            raise ValueError("depth_percentile must be between 0 and 1")
        return int(params.peak_window + params.rank_window + 1)

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["high"].rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = (out["peak"] - data["close"]) / out["peak"].replace(0.0, np.nan)
        out["prior_depth"] = out["drawdown"].shift(1)
        out["depth_rank"] = out["prior_depth"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["price_change"] = data["close"].diff()
        out["prior_gap"] = out["peak"].shift(1) - data["close"].shift(1)
        out["recovery_speed"] = out["price_change"] / out["prior_gap"].where(out["prior_gap"] > 0.0)
        out["recovery_speed"] = out["recovery_speed"].replace([np.inf, -np.inf], np.nan)
        out["recovery_rank"] = out["recovery_speed"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RecoveryParams) -> SignalFrame:
        rank = indicators["recovery_rank"]
        change = indicators["price_change"]
        bullish = (rank >= params.tail_percentile) & (change > 0.0)
        bearish = (rank <= 1.0 - params.tail_percentile) & (change < 0.0)
        deep = indicators["depth_rank"] >= params.depth_percentile
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (up, down, eligible) in enumerate(zip(bullish.to_numpy(), bearish.to_numpy(), deep.to_numpy())):
            if position == 1 and down:
                position = 0
            elif position == -1 and up:
                position = 0
            elif position == 0 and eligible:
                if up:
                    position = 1
                elif down:
                    position = -1
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"recovery_rank": [0.9, 0.9, 0.1, 0.1, 0.9, 0.9], "price_change": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0], "depth_rank": [0.9] * 6}, index=index)
    strategy = GeneratedStrategy()
    params = RecoveryParams()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert (result["size"] > 0.0).all()
    indicators.loc[:, "recovery_rank"] = np.nan
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].eq(0).all()
    assert strategy.warmup_bars(params) >= params.peak_window + params.rank_window


if __name__ == "__main__":
    _self_check()
