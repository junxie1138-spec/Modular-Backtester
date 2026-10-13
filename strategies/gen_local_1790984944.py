from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    volatility_window: int = 20
    breach_multiple: float = 0.8
    streak_length: int = 3
    target_volatility: float = 0.12
    min_size: float = 0.10
    max_size: float = 0.95
    profit_target: float = 0.04
    max_holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local_1790984944"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return int(max(200, params.volatility_window + params.streak_length + 2))

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None)
        out["daily_volatility"] = out["return"].rolling(
            params.volatility_window, min_periods=params.volatility_window
        ).std(ddof=0)
        out["prior_volatility"] = out["daily_volatility"].shift(1)
        out["annual_volatility"] = out["daily_volatility"] * np.sqrt(252.0)
        out["normalized_return"] = (
            out["return"] / out["prior_volatility"].replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["up_breach"] = out["normalized_return"].ge(params.breach_multiple)
        out["down_breach"] = out["normalized_return"].le(-params.breach_multiple)
        out["up_streak"] = out["up_breach"].astype(int).groupby(
            (~out["up_breach"]).cumsum()
        ).cumsum()
        out["down_streak"] = out["down_breach"].astype(int).groupby(
            (~out["down_breach"]).cumsum()
        ).cumsum()
        ready = (
            out["normalized_return"].notna()
            & out["ma200"].notna()
            & out["annual_volatility"].gt(0.0)
            & np.isfinite(out["annual_volatility"])
        )
        out["long_entry"] = (
            ready
            & out["up_streak"].shift(1).ge(params.streak_length)
            & out["up_streak"].eq(0)
            & out["return"].gt(0.0)
            & data["close"].gt(out["ma200"])
        ).fillna(False)
        out["short_entry"] = (
            ready
            & out["down_streak"].shift(1).ge(params.streak_length)
            & out["down_streak"].eq(0)
            & out["return"].lt(0.0)
            & data["close"].lt(out["ma200"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        volatility = indicators["annual_volatility"].to_numpy(dtype=float)
        direction = 0
        entry_bar = -1
        entry_price = np.nan
        position_size = params.min_size

        for i in range(len(data)):
            if direction != 0:
                # Entry decided at the prior close fills at this session's open.
                if i == entry_bar:
                    entry_price = opens[i]
                gain = direction * (closes[i] / entry_price - 1.0)
                elapsed = i - entry_bar + 1
                if gain >= params.profit_target or elapsed >= params.max_holding_bars:
                    direction = 0
            elif np.isfinite(volatility[i]) and volatility[i] > 0.0:
                candidate = 1 if longs[i] else (-1 if shorts[i] else 0)
                if candidate != 0:
                    direction = candidate
                    entry_bar = i + 1
                    entry_price = np.nan
                    position_size = float(np.clip(
                        params.target_volatility / volatility[i],
                        params.min_size,
                        params.max_size,
                    ))
            signals[i] = direction
            sizes[i] = position_size

        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame({
        "open": [100.0] * 6,
        "high": [104.0] * 6,
        "low": [96.0] * 6,
        "close": [100.0, 103.0, 100.0, 100.0, 97.0, 100.0],
        "volume": [1000.0] * 6,
    }, index=index)
    indicators = pd.DataFrame({
        "long_entry": [True, False, False, False, False, False],
        "short_entry": [False, False, False, True, False, False],
        "annual_volatility": [0.24] * 6,
    }, index=index)
    strategy = GeneratedStrategy()
    params = StreakParams(profit_target=0.02, max_holding_bars=2)
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0]
    assert np.isclose(result["size"].iloc[1], 0.5)
    data["close"] = 100.0
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1]
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
