from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    streak_days: int = 2
    pressure_window: int = 18
    vol_window: int = 20
    pressure_threshold: float = 0.08
    target_vol: float = 0.12
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791334306"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.pressure_window, params.vol_window + 1,
                       params.streak_days + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.streak_days, params.pressure_window, params.vol_window) < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not (0.0 <= params.pressure_threshold < 1.0):
            raise ValueError("pressure_threshold must be in [0, 1).")
        if not (0.0 < params.target_vol and 0.0 < params.max_size <= 1.0):
            raise ValueError("Positive target_vol and max_size in (0, 1] required.")
        out = pd.DataFrame(index=data.index)
        out["range"] = (data["high"] - data["low"]).where(
            data["high"] > data["low"])
        out["auction_efficiency"] = (
            (data["close"] - data["open"]) / out["range"]
        ).replace([np.inf, -np.inf], np.nan)
        out["efficiency_change"] = out["auction_efficiency"].diff()
        rising = out["efficiency_change"].gt(0)
        falling = out["efficiency_change"].lt(0)
        out["rising_streak"] = rising.astype(int).groupby(
            (~rising).cumsum()).cumsum()
        out["falling_streak"] = falling.astype(int).groupby(
            (~falling).cumsum()).cumsum()
        out["close_location"] = (
            (2.0 * data["close"] - data["high"] - data["low"]) / out["range"]
        ).replace([np.inf, -np.inf], np.nan)
        out["dollar_volume"] = (data["close"] * data["volume"]).where(
            (data["close"] > 0) & (data["volume"] > 0))
        out["weighted_location"] = out["close_location"] * out["dollar_volume"]
        out["pressure_numerator"] = out["weighted_location"].rolling(
            params.pressure_window).sum()
        out["turnover_sum"] = out["dollar_volume"].rolling(
            params.pressure_window).sum()
        out["capital_pressure"] = out["pressure_numerator"] / out[
            "turnover_sum"].where(out["turnover_sum"] > 0)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["annual_vol"] = out["return"].rolling(
            params.vol_window).std(ddof=0) * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        long_entry = (
            indicators["rising_streak"].ge(params.streak_days)
            & indicators["capital_pressure"].gt(params.pressure_threshold)
        )
        short_entry = (
            indicators["falling_streak"].ge(params.streak_days)
            & indicators["capital_pressure"].lt(-params.pressure_threshold)
        )
        candidate = np.select([long_entry, short_entry], [1, -1], default=0)
        ready = indicators["annual_vol"].notna().to_numpy()
        state = 0
        raw = np.zeros(len(data), dtype=int)
        for i, direction in enumerate(candidate):
            if state == 0:
                if ready[i]:
                    state = int(direction)
            elif direction == -state:
                # Exit to flat; an opposite entry can occur on the next decision.
                state = 0
            raw[i] = state
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = (
            params.target_vol / indicators["annual_vol"].clip(lower=0.01)
        ).clip(lower=0.01, upper=params.max_size).shift(1).fillna(
            params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "rising_streak": [2, 0, 0, 0, 0, 0, 2, 2, 2],
        "falling_streak": [0, 0, 2, 2, 2, 0, 0, 0, 0],
        "capital_pressure": [0.2, 0.0, 0.2, -0.2, -0.2, 0.0, 0.2, 0.2, 0.2],
        "annual_vol": [0.24] * 9,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0, 1]
    assert np.isclose(result["size"].iloc[1], 0.5)
    assert result["size"].gt(0).all()
