from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    level_window: int = 20
    slope_window: int = 8
    innovation_window: int = 3
    volatility_window: int = 20
    percentile_window: int = 80
    tail_quantile: float = 0.80
    profit_target: float = 0.025
    max_hold_bars: int = 10
    size: float = 0.95


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791230802"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.level_window, params.volatility_window + 1,
                       params.slope_window + params.innovation_window + 1)
                   + params.percentile_window + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.level_window, params.slope_window, params.innovation_window,
               params.volatility_window, params.percentile_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 0.5 < params.tail_quantile < 1.0:
            raise ValueError("tail_quantile must lie between 0.5 and 1")
        log_price = np.log(data["close"].where(data["close"] > 0))
        returns = log_price.diff()
        level = log_price.rolling(params.level_window).mean()
        slope = returns.rolling(params.slope_window).mean()
        volatility = returns.rolling(params.volatility_window).std(ddof=0)
        tension = (log_price - level) / volatility.where(volatility > 0)
        innovation = (returns - slope.shift(1)).rolling(params.innovation_window).mean()
        tension_threshold = tension.abs().rolling(params.percentile_window).quantile(params.tail_quantile).shift(1)
        innovation_threshold = innovation.abs().rolling(params.percentile_window).quantile(params.tail_quantile).shift(1)
        return pd.DataFrame({
            "log_price": log_price,
            "returns": returns,
            "level": level,
            "slope": slope,
            "volatility": volatility,
            "tension": tension,
            "innovation": innovation,
            "tension_threshold": tension_threshold,
            "innovation_threshold": innovation_threshold,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if params.max_hold_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("Holding period and profit target must be positive")
        if not np.isfinite(params.size) or not 0 < params.size <= 1:
            raise ValueError("size must lie in (0, 1]")
        tension = indicators["tension"]
        innovation = indicators["innovation"]
        tension_threshold = indicators["tension_threshold"]
        innovation_threshold = indicators["innovation_threshold"]
        ready = np.isfinite(indicators[["tension", "innovation", "tension_threshold", "innovation_threshold"]]).all(axis=1)
        longs = ready & (tension < -tension_threshold) & (innovation > innovation_threshold)
        shorts = ready & (tension > tension_threshold) & (innovation < -innovation_threshold)
        entries = np.select([longs, shorts], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at today's open.
                if held == 0:
                    entry_price = opens[i]
                held += 1
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = position * (closes[i] / entry_price - 1.0) if valid_entry else np.nan
                # Closing decisions execute at the following open after the shift.
                if not valid_entry or gain >= params.profit_target or held >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
                    held = 0
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(entries[i])
                held = 0
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0,
                         "close": [100.0, 100.0, 103.0, 100.0, 100.0],
                         "volume": 1000.0}, index=index)
    features = pd.DataFrame({"tension": [-2.0, 0.0, 0.0, 0.0, 0.0],
                             "innovation": [2.0, 0.0, 0.0, 0.0, 0.0],
                             "tension_threshold": 1.0,
                             "innovation_threshold": 1.0}, index=index)
    strategy = object.__new__(GeneratedStrategy)
    result = strategy.generate_signals(data, features, None, StrategyParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, features, None, StrategyParams(max_hold_bars=2)).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0]
    features["tension"] *= -1
    features["innovation"] *= -1
    data.loc[index[2], "close"] = 97.0
    result = strategy.generate_signals(data, features, None, StrategyParams()).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
