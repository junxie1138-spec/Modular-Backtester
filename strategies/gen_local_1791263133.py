from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapFailureParams:
    atr_window: int = 14
    volume_window: int = 30
    min_gap_atr: float = 0.6
    min_relative_volume: float = 1.5
    min_reclaim_atr: float = 0.2
    stop_atr: float = 2.5
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapFailureParams]):
    strategy_id = "gen_local_1791263133"

    @classmethod
    def params_type(cls) -> type[GapFailureParams]:
        return GapFailureParams

    def warmup_bars(self, params: GapFailureParams) -> int:
        if min(params.atr_window, params.volume_window, params.holding_bars) < 1:
            raise ValueError("Windows and holding_bars must be positive.")
        if min(params.min_gap_atr, params.min_reclaim_atr, params.stop_atr) <= 0 or params.min_relative_volume <= 1:
            raise ValueError("Thresholds must be positive and relative volume must exceed one.")
        return int(max(params.atr_window, params.volume_window) + 1)

    def indicators(self, data: pd.DataFrame, params: GapFailureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["previous_close"] = previous_close
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["prior_atr"] = out["atr"].shift(1).where(lambda x: x > 0)
        out["prior_volume"] = data["volume"].shift(1).rolling(params.volume_window, min_periods=params.volume_window).mean()
        out["relative_volume"] = data["volume"] / out["prior_volume"].where(out["prior_volume"] > 0)
        out["gap_atr"] = (data["open"] - previous_close) / out["prior_atr"]
        out["reclaim_atr"] = -np.sign(out["gap_atr"]) * (data["close"] - previous_close) / out["prior_atr"]
        out["close_location"] = (data["close"] - data["low"]) / (data["high"] - data["low"]).where(data["high"] > data["low"])
        valid = (
            out["gap_atr"].abs().ge(params.min_gap_atr)
            & out["relative_volume"].ge(params.min_relative_volume)
            & out["reclaim_atr"].ge(params.min_reclaim_atr)
            & out["atr"].gt(0)
        )
        long_entry = valid & out["gap_atr"].lt(0) & out["close_location"].ge(0.75)
        short_entry = valid & out["gap_atr"].gt(0) & out["close_location"].le(0.25)
        out["direction"] = np.select([long_entry, short_entry], [1, -1], default=0).astype(int)
        out["strength"] = (
            (out["gap_atr"].abs() / params.min_gap_atr)
            * (out["relative_volume"] / params.min_relative_volume)
            * (out["reclaim_atr"] / params.min_reclaim_atr)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapFailureParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.full(n, 0.25, dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        position_size = 0.25
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        direction = indicators["direction"].fillna(0).to_numpy(dtype=int)
        strength = indicators["strength"].fillna(0).to_numpy(dtype=float)
        for i in range(n):
            if side:
                if i == entry_bar:
                    entry_price = opening[i]
                invalid = not (np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closing[i]))
                stopped = side * (closing[i] - entry_price) < -params.stop_atr * entry_atr
                expired = i - entry_bar + 1 >= params.holding_bars
                if invalid or stopped or expired:
                    side = 0
                signals[i] = side
                sizes[i] = position_size
                continue
            if direction[i] and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(strength[i]):
                side = int(direction[i])
                entry_bar = i + 1
                entry_atr = atr[i]
                entry_price = np.nan
                position_size = float(0.25 + 0.70 * np.clip((strength[i] - 1.0) / 3.0, 0.0, 1.0))
                signals[i] = side
                sizes[i] = position_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GapFailureParams(stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=22, freq="B")
    bars = pd.DataFrame({"open": 100.0, "high": 111.0, "low": 89.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"atr": 1.0, "direction": 0, "strength": 4.0}, index=index)
    derived.iloc[0, derived.columns.get_loc("direction")] = 1
    result = strategy.generate_signals(bars, derived, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 3
    assert np.isclose(result["size"].iloc[1], 0.95)
    bars.loc[index[2], "close"] = 110.0
    bars.loc[index[3], "close"] = 97.0
    derived.loc[index[4], "direction"] = -1
    derived.loc[index[4], "strength"] = 1.0
    bars.loc[index[6], "close"] = 90.0
    bars.loc[index[7], "close"] = 103.0
    result = strategy.generate_signals(bars, derived, None, params).data
    assert result["signal"].iloc[:9].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    assert np.isclose(result["size"].iloc[5], 0.25)
    assert strategy.indicators(bars, params)["direction"].eq(0).all()
