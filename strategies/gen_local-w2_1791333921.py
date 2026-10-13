from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PressureRankParams:
    rank_window: int = 63
    pressure_window: int = 3
    volatility_window: int = 20
    holding_bars: int = 4
    divergence_threshold: float = 0.45
    target_volatility: float = 0.10
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[PressureRankParams]):
    strategy_id = "gen_local-w2_1791333921"

    @classmethod
    def params_type(cls) -> type[PressureRankParams]:
        return PressureRankParams

    def warmup_bars(self, params: PressureRankParams) -> int:
        return int(max(params.rank_window + params.pressure_window,
                       params.volatility_window + 1))

    def indicators(self, data: pd.DataFrame, params: PressureRankParams) -> pd.DataFrame:
        if min(params.rank_window, params.pressure_window, params.volatility_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        out = pd.DataFrame(index=data.index)
        out["daily_return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["cumulative_return"] = data["close"].pct_change(params.pressure_window, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["daily_range"] = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["close_location"] = ((2.0 * data["close"] - data["high"] - data["low"]) / out["daily_range"]).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out["pressure"] = out["close_location"].rolling(params.pressure_window, min_periods=params.pressure_window).mean()
        out["pressure_rank"] = out["pressure"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["return_rank"] = out["cumulative_return"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["rank_gap"] = out["pressure_rank"] - out["return_rank"]
        out["volatility"] = out["daily_return"].rolling(params.volatility_window, min_periods=params.volatility_window).std(ddof=0) * np.sqrt(252.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: PressureRankParams) -> SignalFrame:
        if params.holding_bars < 1 or not 0.0 < params.divergence_threshold < 1.0:
            raise ValueError("Invalid holding period or divergence threshold")
        if params.target_volatility <= 0.0 or not 0.05 <= params.max_size <= 1.0:
            raise ValueError("Invalid volatility target or size cap")
        gap = indicators["rank_gap"].to_numpy(dtype=float)
        pressure = indicators["pressure_rank"].to_numpy(dtype=float)
        volatility = indicators["volatility"].to_numpy(dtype=float)
        valid = np.isfinite(gap) & np.isfinite(pressure) & np.isfinite(volatility) & (volatility > 0.0)
        candidates = np.zeros(len(data), dtype=int)
        candidates[valid & (gap >= params.divergence_threshold) & (pressure >= 0.65)] = 1
        candidates[valid & (gap <= -params.divergence_threshold) & (pressure <= 0.35)] = -1
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.05, dtype=float)
        side, age, entry_size = 0, 0, 0.05
        for i in range(len(data)):
            if side:
                age += 1
                if age >= params.holding_bars:
                    side, age = 0, 0
            elif candidates[i]:
                side = int(candidates[i])
                age = 0
                entry_size = float(np.clip(params.target_volatility / volatility[i] * abs(gap[i]), 0.05, params.max_size))
            signals[i] = side
            sizes[i] = entry_size if side else 0.05
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = PressureRankParams()
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    for direction in (1, -1):
        gap = np.full(len(data), 0.8 * direction)
        gap[1:4] *= -1
        derived = pd.DataFrame({"rank_gap": gap, "pressure_rank": np.where(gap > 0, 0.9, 0.1), "volatility": 0.20}, index=index)
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].iloc[:6].tolist() == [0] + [direction] * 4 + [0]
        assert np.allclose(result["size"].iloc[1:5], 0.4)
        derived.iloc[:] = np.nan
        result = strategy.generate_signals(data, derived, None, params).data
        assert result["signal"].eq(0).all() and result["size"].gt(0).all()
