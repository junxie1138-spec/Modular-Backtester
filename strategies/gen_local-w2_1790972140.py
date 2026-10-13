from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    level_span: int = 5
    slope_span: int = 3
    expectation_span: int = 20
    scale_window: int = 40
    innovation_threshold: float = 0.8
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1790972140"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(params.level_span + params.slope_span + params.expectation_span + params.scale_window + 10)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.level_span, params.slope_span, params.expectation_span, params.scale_window) < 2:
            raise ValueError("Indicator spans and windows must be at least two.")
        if params.innovation_threshold <= 0 or params.profit_target <= 0 or params.max_hold_bars < 1:
            raise ValueError("Threshold, target, and holding limit must be positive.")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        out["latent_level"] = out["log_close"].ewm(span=params.level_span, adjust=False, min_periods=params.level_span).mean()
        out["level_change"] = out["latent_level"].diff()
        out["latent_slope"] = out["level_change"].ewm(span=params.slope_span, adjust=False, min_periods=params.slope_span).mean()
        out["expected_slope"] = out["latent_slope"].ewm(span=params.expectation_span, adjust=False, min_periods=params.expectation_span).mean().shift(1)
        out["innovation"] = out["latent_slope"] - out["expected_slope"]
        out["innovation_scale"] = out["innovation"].rolling(params.scale_window, min_periods=params.scale_window).std(ddof=0).shift(1)
        out["innovation_score"] = (out["innovation"] / out["innovation_scale"].where(out["innovation_scale"] > 1e-12)).replace([np.inf, -np.inf], np.nan)
        long_setup = (out["latent_slope"] > 0) & (out["expected_slope"] < 0) & (out["innovation_score"] >= params.innovation_threshold)
        short_setup = (out["latent_slope"] < 0) & (out["expected_slope"] > 0) & (out["innovation_score"] <= -params.innovation_threshold)
        out["candidate"] = np.select([long_setup, short_setup], [1, -1], default=0).astype(int)
        previous = out["candidate"].shift(1).fillna(0)
        out["confirmed"] = out["candidate"].where((out["candidate"] != 0) & (out["candidate"] == previous), 0).astype(int)
        out["entry_direction"] = out["confirmed"].where(out["confirmed"] != out["confirmed"].shift(1).fillna(0), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        direction = 0
        pending_entry = False
        entry_price = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if pending_entry:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    raise ValueError("Entry open must be finite and positive.")
                entry_price = opens[i]
                entry_bar = i
                pending_entry = False
            if direction != 0:
                held_bars = i - entry_bar + 1
                gain = direction * (closes[i] / entry_price - 1.0)
                target_hit = np.isfinite(gain) and gain >= params.profit_target
                if target_hit or held_bars >= params.max_hold_bars:
                    direction = 0
                decisions[i] = direction
                continue
            if entries[i] != 0:
                direction = int(entries[i])
                pending_entry = True
            decisions[i] = direction
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0, "close": [100.0, 100.0, 104.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry_direction": [0, 1, 0, 0, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, StrategyParams(max_hold_bars=2)).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0]
    indicators["entry_direction"] = [0, -1, 0, 0, 0, 0]
    data.loc[index[2], "close"] = 96.0
    result = strategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result["signal"].tolist() == [0, 0, -1, 0, 0, 0]
