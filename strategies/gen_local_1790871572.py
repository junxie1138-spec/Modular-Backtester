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
    volatility_window: int = 20
    channel_window: int = 10
    innovation_threshold: float = 0.35
    profit_target: float = 0.012
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790871572"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(3 * (params.level_span + params.slope_span), params.volatility_window + 1, params.channel_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.level_span, params.slope_span, params.volatility_window, params.channel_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if params.profit_target <= 0 or params.hold_bars not in (1, 2):
            raise ValueError("Use a positive profit target and a one- or two-bar holding period")
        result = pd.DataFrame(index=data.index)
        result["log_close"] = np.log(data["close"].where(data["close"] > 0))
        result["return"] = result["log_close"].diff()
        result["level"] = result["log_close"].ewm(span=params.level_span, adjust=False, min_periods=params.level_span).mean()
        result["slope"] = result["level"].diff().ewm(span=params.slope_span, adjust=False, min_periods=params.slope_span).mean()
        result["forecast_error"] = result["return"] - result["slope"].shift(1)
        result["volatility"] = result["return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0)
        result["innovation"] = result["forecast_error"] / result["volatility"].shift(1).replace(0.0, np.nan)
        result["slope_change"] = result["slope"].diff()
        result["channel_low"] = data["low"].rolling(params.channel_window, min_periods=params.channel_window).min()
        result["channel_high"] = data["high"].rolling(params.channel_window, min_periods=params.channel_window).max()
        result["position"] = (data["close"] - result["channel_low"]) / (result["channel_high"] - result["channel_low"]).replace(0.0, np.nan)
        result["previous_position"] = result["position"].shift(1)
        slope_agreement = (result["innovation"] > params.innovation_threshold) & (result["slope_change"] > 0)
        position_agreement = (result["previous_position"] <= 0.5) & (result["position"] > 0.5)
        result["entry"] = (slope_agreement & position_agreement).fillna(False)
        return result.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if params.profit_target <= 0 or params.hold_bars not in (1, 2):
            raise ValueError("Use a positive profit target and a one- or two-bar holding period")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        decisions = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        entry_price = 0.0
        held = 0
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    active = True
                    entry_price = opens[i]
                    held = 0
            if active:
                held += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held >= params.hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                decisions[i] = 1
                pending = True
        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100.0] * 5, "close": [100.0] * 5}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 102.0
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 0, 0, 0]
