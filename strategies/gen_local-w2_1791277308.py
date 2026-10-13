from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    atr_period: int = 14
    season_window: int = 63
    shock_ratio: float = 1.0


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791277308"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.atr_period, params.season_window) + 1

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.atr_period < 2 or params.season_window < 20 or not np.isfinite(params.shock_ratio) or params.shock_ratio <= 0:
            raise ValueError("Invalid lookbacks or shock threshold")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["return"] = data["close"].div(previous_close.replace(0, np.nan)).sub(1).replace([np.inf, -np.inf], np.nan)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["prior_atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean().shift(1)
        out["shock_ratio"] = out["true_range"].div(out["prior_atr"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["fade_direction"] = -np.sign(out["return"].fillna(0))
        out["weekday"] = data.index.dayofweek
        out["previous_weekday"] = out["weekday"].shift(1)
        out["seasonal_edge"] = np.nan
        minimum_samples = max(3, params.season_window // 10)
        for weekday in range(5):
            # Today's realized return labels yesterday's decision weekday.
            sample = out["return"].where(out["previous_weekday"].eq(weekday))
            name = f"weekday_{weekday}_edge"
            out[name] = sample.rolling(params.season_window, min_periods=minimum_samples).mean()
            mask = out["weekday"].eq(weekday)
            out.loc[mask, "seasonal_edge"] = out.loc[mask, name]
        out.loc[out.index[:self.warmup_bars(params)], "seasonal_edge"] = np.nan
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        shock = indicators["shock_ratio"].ge(params.shock_ratio)
        long_entry = shock & indicators["fade_direction"].eq(1) & indicators["seasonal_edge"].gt(0)
        short_entry = shock & indicators["fade_direction"].eq(-1) & indicators["seasonal_edge"].lt(0)
        evidence = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(evidence):
            if position != 0 and candidate == -position:
                position = 0
            elif position == 0 and candidate != 0:
                position = int(candidate)
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame(index=index)
    evidence = np.array([1, 0, -1, -1, 0, 1, 1])
    indicators = pd.DataFrame({
        "shock_ratio": np.full(7, 2.0),
        "fade_direction": evidence,
        "seasonal_edge": evidence.astype(float),
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    indicators["seasonal_edge"] = np.nan
    assert strategy.generate_signals(data, indicators, ctx, params).data["signal"].eq(0).all()
