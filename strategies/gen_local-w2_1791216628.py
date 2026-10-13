from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 12
    percentile_window: int = 126
    entry_percentile: float = 0.85
    volatility_window: int = 20
    target_volatility: float = 0.12
    atr_window: int = 14
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791216628"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return int(max(params.graph_window + params.percentile_window + 1,
                       params.volatility_window + 1, params.atr_window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.graph_window, params.percentile_window,
               params.volatility_window, params.atr_window) < 2:
            raise ValueError("Window lengths must be at least two")
        if not 0.5 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must lie between 0.5 and 1")
        if params.target_volatility <= 0 or params.stop_atr <= 0:
            raise ValueError("Volatility target and stop multiplier must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        out["log_return"] = out["log_close"].diff()
        out["annual_volatility"] = out["log_return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0) * np.sqrt(252.0)
        previous = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs()
        ], axis=1).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        upper_span = pd.Series(0.0, index=data.index)
        lower_span = pd.Series(0.0, index=data.index)
        minimum = pd.Series(np.inf, index=data.index)
        maximum = pd.Series(-np.inf, index=data.index)
        valid = out["log_close"].notna()
        for lag in range(1, params.graph_window + 1):
            past = out["log_close"].shift(lag)
            slope = (out["log_close"] - past) / lag
            valid = valid & slope.notna()
            upper_span += ((slope < minimum) & valid).astype(float) * lag
            lower_span += ((slope > maximum) & valid).astype(float) * lag
            minimum = np.minimum(minimum, slope)
            maximum = np.maximum(maximum, slope)
        out["upper_span"] = upper_span.where(valid)
        out["lower_span"] = lower_span.where(valid)
        out["topology"] = ((out["upper_span"] - out["lower_span"]) /
                           (out["upper_span"] + out["lower_span"]))
        history = out["topology"].shift(1).rolling(
            params.percentile_window, min_periods=params.percentile_window
        )
        out["upper_threshold"] = history.quantile(params.entry_percentile)
        out["lower_threshold"] = history.quantile(1.0 - params.entry_percentile)
        out["target_size"] = (params.target_volatility /
            out["annual_volatility"].replace(0.0, np.nan)
        ).clip(0.05, 0.95).fillna(0.05)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        required = ["topology", "upper_threshold", "lower_threshold",
                    "atr", "annual_volatility"]
        ready = indicators[required].notna().all(axis=1)
        ready &= (indicators["atr"] > 0) & (indicators["annual_volatility"] > 0)
        long_entry = ready & (indicators["topology"] > indicators["upper_threshold"])
        short_entry = ready & (indicators["topology"] < indicators["lower_threshold"])
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        target_size = indicators["target_size"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        position = 0
        entry_bar = -1
        stop = np.nan
        entry_size = 0.05
        for i in range(len(data)):
            previous_target = desired[i - 1] if i else 0
            if previous_target == 0:
                position = 0
            elif position == 0:
                position = int(previous_target)
                entry_bar = i
                entry_size = sizes[i - 1]
                stop = opens[i] - position * params.stop_atr * atr[i - 1]
            if position:
                sizes[i] = entry_size
                stopped = (not np.isfinite(closes[i]) or
                           not np.isfinite(stop) or
                           (position == 1 and closes[i] < stop) or
                           (position == -1 and closes[i] > stop))
                expired = i - entry_bar >= 1
                desired[i] = 0 if stopped or expired else position
            elif candidate[i] != 0:
                desired[i] = int(candidate[i])
                sizes[i] = float(target_size[i])
        df = pd.DataFrame({"decision": desired, "signal": desired,
                           "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": [100.0] * 6,
                         "high": [101.0] * 6,
                         "low": [96.0] * 6,
                         "close": [100.0, 97.0, 100.0, 100.0, 100.0, 100.0],
                         "volume": [1000.0] * 6}, index=index)
    indicators = pd.DataFrame({"topology": [1.0] * 6,
                               "upper_threshold": [0.5] * 6,
                               "lower_threshold": [-0.5] * 6,
                               "atr": [1.0] * 6,
                               "annual_volatility": [0.2] * 6,
                               "target_size": [0.6] * 6}, index=index)
    params = Params()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 1, 1, 0]
    assert result["size"].gt(0).all()
    indicators["topology"] = -1.0
    data.loc[index[1], "close"] = 103.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, -1, 0, -1, -1, 0]
