from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    rank_window: int = 60
    coupling_window: int = 20
    coupling_threshold: float = 0.15
    tail_fraction: float = 0.20
    body_floor: float = 0.35
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790854728"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.rank_window, params.coupling_window + 1) + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.rank_window < 2 or params.coupling_window < 3:
            raise ValueError("Lookback windows are too short.")
        if not 0 < params.coupling_threshold < 1:
            raise ValueError("coupling_threshold must be between zero and one.")
        if not 0 < params.tail_fraction < 0.5 or not 0 <= params.body_floor <= 1:
            raise ValueError("Invalid pressure thresholds.")
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be 3, 4, or 5.")

        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        valid_open = data["open"].where(data["open"] > 0)
        daily_range = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["overnight_return"] = data["open"] / previous_close - 1.0
        out["intraday_return"] = data["close"] / valid_open - 1.0
        out["pressure"] = (data["close"] - data["open"]) / daily_range
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out["coupling"] = out["overnight_return"].rolling(
            params.coupling_window, min_periods=params.coupling_window
        ).corr(out["intraday_return"]).clip(-1.0, 1.0)
        out["pressure_rank"] = out["pressure"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(method="average", pct=True)
        out["regime"] = np.select(
            [out["coupling"] >= params.coupling_threshold,
             out["coupling"] <= -params.coupling_threshold],
            [1, -1], default=0
        ).astype(int)
        out["extreme"] = np.select(
            [(out["pressure_rank"] >= 1.0 - params.tail_fraction) &
             (out["pressure"] >= params.body_floor),
             (out["pressure_rank"] <= params.tail_fraction) &
             (out["pressure"] <= -params.body_floor)],
            [1, -1], default=0
        ).astype(int)
        out["setup"] = out["regime"] * out["extreme"]
        confirmed = (
            out["setup"].ne(0) &
            out["setup"].eq(out["setup"].shift(1)) &
            out["regime"].eq(out["regime"].shift(1))
        )
        out["confirmed"] = out["setup"].where(confirmed, 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if params.holding_bars not in (3, 4, 5):
            raise ValueError("holding_bars must be 3, 4, or 5.")
        entries = indicators["confirmed"].fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(entries):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = i + params.holding_bars
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame(
        {"confirmed": [0, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 0]}, index=index
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
