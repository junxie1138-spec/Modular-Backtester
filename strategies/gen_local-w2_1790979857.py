from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    history: int = 126
    congestion_window: int = 10
    streak_quantile: float = 0.85
    congestion_quantile: float = 0.50
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1790979857"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    @staticmethod
    def warmup_bars(params: StreakParams) -> int:
        return int(params.history + params.congestion_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.history < 2 or params.congestion_window < 1:
            raise ValueError("Lookbacks must be positive and history must be at least two")
        if not 0.0 < params.streak_quantile < 1.0 or not 0.0 < params.congestion_quantile < 1.0:
            raise ValueError("Quantiles must lie strictly between zero and one")
        if params.profit_target <= 0.0 or params.max_hold < 1:
            raise ValueError("Profit target and holding period must be positive")
        out = pd.DataFrame(index=data.index)
        change = data["close"].diff()
        out["direction"] = np.sign(change).fillna(0).astype(int)
        groups = out["direction"].ne(out["direction"].shift()).cumsum()
        out["streak"] = out.groupby(groups).cumcount().add(1).where(out["direction"].ne(0), 0)
        out["streak_threshold"] = out["streak"].shift(1).rolling(params.history, min_periods=params.history).quantile(params.streak_quantile)
        out["range"] = (data["high"] - data["low"]).clip(lower=0)
        out["overlap"] = (pd.concat([data["high"], data["high"].shift(1)], axis=1).min(axis=1, skipna=False) - pd.concat([data["low"], data["low"].shift(1)], axis=1).max(axis=1, skipna=False)).clip(lower=0)
        out["overlap_scale"] = pd.concat([out["range"], out["range"].shift(1)], axis=1).min(axis=1, skipna=False).replace(0.0, np.nan)
        out["overlap_ratio"] = (out["overlap"] / out["overlap_scale"]).clip(0.0, 1.0)
        out["congestion"] = out["overlap_ratio"].rolling(params.congestion_window, min_periods=params.congestion_window).mean()
        out["congestion_threshold"] = out["congestion"].shift(1).rolling(params.history, min_periods=params.history).quantile(params.congestion_quantile)
        out["rare_streak"] = out["streak"].ge(3) & out["streak"].gt(out["streak_threshold"])
        out["fresh_streak"] = out["rare_streak"] & ~out["rare_streak"].shift(1, fill_value=False)
        ready = out[["streak_threshold", "congestion", "congestion_threshold"]].notna().all(axis=1)
        regime = pd.Series(np.where(out["congestion"].gt(out["congestion_threshold"]), -1, 1), index=data.index)
        out["entry_direction"] = (out["direction"] * regime).where(ready & out["fresh_streak"], 0).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StreakParams) -> SignalFrame:
        if params.profit_target <= 0.0 or params.max_hold < 1:
            raise ValueError("Profit target and holding period must be positive")
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position != 0:
                if i == entry_bar:
                    # The prior close's decision fills at this session's open.
                    entry_price = opens[i]
                age = i - entry_bar + 1
                valid_price = np.isfinite(entry_price) and entry_price > 0.0
                gain = position * (closes[i] / entry_price - 1.0) if valid_price and np.isfinite(closes[i]) else np.nan
                if not valid_price or gain >= params.profit_target or age >= params.max_hold:
                    position = 0
                decisions[i] = position
                continue
            if entries[i] != 0 and i + 1 < len(data) and np.isfinite(closes[i]) and closes[i] > 0.0:
                position = int(entries[i])
                entry_bar = i + 1
                entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    for direction in (1, -1):
        sample = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0, "close": [100.0, 100.0, 100.0 + direction * 3.0, 100.0, 100.0, 100.0], "volume": 1000.0}, index=index)
        derived = pd.DataFrame({"entry_direction": [direction, 0, 0, 0, 0, 0]}, index=index)
        for settings in (StreakParams(profit_target=0.02, max_hold=8), StreakParams(profit_target=0.50, max_hold=2)):
            result = GeneratedStrategy.generate_signals(sample, derived, None, settings).data
            assert result["signal"].tolist() == [0, direction, direction, 0, 0, 0]
            assert result["size"].gt(0.0).all()
