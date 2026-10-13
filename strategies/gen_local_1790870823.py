from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    lookback: int = 252
    atr_period: int = 14
    extreme_quantile: float = 0.8
    hold_days: int = 4
    stop_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790870823"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.lookback + 2, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (params.lookback >= 60 and params.atr_period >= 1
                and 0.5 < params.extreme_quantile < 1.0
                and 3 <= params.hold_days <= 5
                and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid range-seasonality parameters")
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["range"] = span
        out["midpoint"] = (data["high"] + data["low"]) / 2.0
        out["expansion"] = (span / span.shift(1)).replace([np.inf, -np.inf], np.nan)
        prior = out["expansion"].shift(1)
        out["pooled_median"] = prior.rolling(params.lookback, min_periods=params.lookback).median()
        out["weekday_median"] = np.nan
        out["weekday_threshold"] = np.nan
        weekdays = data.index.dayofweek
        minimum = max(5, params.lookback // 10)
        for weekday in range(5):
            mask = weekdays == weekday
            history = out["expansion"].where(mask).shift(1).rolling(
                params.lookback, min_periods=minimum
            )
            out.loc[mask, "weekday_median"] = history.median().loc[mask]
            out.loc[mask, "weekday_threshold"] = history.quantile(params.extreme_quantile).loc[mask]
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            span, (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs()
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        ready = indicators[[
            "expansion", "pooled_median", "weekday_median",
            "weekday_threshold", "midpoint", "atr"
        ]].notna().all(axis=1)
        event = (ready
                 & (indicators["weekday_median"] < indicators["pooled_median"])
                 & (indicators["expansion"] > indicators["weekday_threshold"])
                 & (indicators["atr"] > 0))
        direction = np.sign(indicators["midpoint"] - data["close"]).fillna(0)
        candidates = direction.where(event, 0).astype(int).to_numpy()
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_atr = 0.0
        stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    # The prior close's order fills at this bar's open.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                stopped = np.isfinite(closes[i]) and position * (closes[i] - stop) < 0
                if stopped or i - entry_bar + 1 >= params.hold_days:
                    position = 0
            elif i >= self.warmup_bars(params) - 1 and candidates[i]:
                position = int(candidates[i])
                entry_bar = i + 1
                entry_atr = atr[i]
            raw[i] = position
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # One deterministic check: delayed fills, both fixed stops, four-day exit.
    p = StrategyParams(lookback=60)
    index = pd.bdate_range("2020-01-01", periods=80)
    bars = pd.DataFrame({
        "open": 100.0, "high": 110.0, "low": 90.0,
        "close": 100.0, "volume": 1000.0
    }, index=index)
    features = pd.DataFrame({
        "expansion": 0.0, "pooled_median": 1.0, "weekday_median": 0.8,
        "weekday_threshold": 2.0, "midpoint": 100.0, "atr": 2.0
    }, index=index)
    features.loc[index[[61, 65, 69]], "expansion"] = 3.0
    bars.loc[index[[61, 69]], "close"] = 99.0
    bars.loc[index[65], "close"] = 101.0
    bars.loc[index[62], "open"] = 105.0
    bars.loc[index[[62, 63]], "close"] = [104.0, 101.0]
    bars.loc[index[66], "open"] = 95.0
    bars.loc[index[[66, 67]], "close"] = [96.0, 99.0]
    strategy = GeneratedStrategy()
    context = StrategyContext("SPY", "1d", strategy.warmup_bars(p))
    result = strategy.generate_signals(bars, features, context, p).data
    expected = np.zeros(80, dtype=int)
    expected[62:64] = 1
    expected[66:68] = -1
    expected[70:74] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()
