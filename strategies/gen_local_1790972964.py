from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_order: float = 0.45
    frac_window: int = 48
    rank_window: int = 63
    tail_rank: float = 0.15
    confirmation_rank: float = 0.55
    arm_bars: int = 20
    atr_window: int = 14
    stop_atr: float = 2.5
    hold_bars: int = 18

    def __post_init__(self):
        if not (0.0 < self.frac_order < 1.0):
            raise ValueError("frac_order must be between zero and one")
        if not (0.0 < self.tail_rank < 0.5 < self.confirmation_rank < 1.0 - self.tail_rank):
            raise ValueError("rank thresholds must define distinct tails and center")
        if min(self.frac_window, self.rank_window, self.atr_window) < 2:
            raise ValueError("indicator windows must be at least two")
        if min(self.arm_bars, self.hold_bars) < 1 or not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("holding windows and stop_atr must be positive")


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790972964"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.frac_window + params.rank_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        weights = np.empty(params.frac_window, dtype=float)
        weights[0] = 1.0
        for lag in range(1, params.frac_window):
            weights[lag] = -weights[lag - 1] * (params.frac_order - lag + 1) / lag
        # ponytail: fixed-width fractional memory; increase frac_window if truncation dominates.
        out["frac_close"] = np.nan
        if len(data) >= params.frac_window:
            out.iloc[params.frac_window - 1:, out.columns.get_loc("frac_close")] = np.convolve(
                close.to_numpy(), weights, mode="valid"
            )
        out["rank"] = out["frac_close"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - out["previous_close"]).abs(),
                (data["low"] - out["previous_close"]).abs(),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["long_zone"] = out["rank"].gt(params.confirmation_rank)
        out["short_zone"] = out["rank"].lt(1.0 - params.confirmation_rank)
        out["long_confirmed"] = out["long_zone"] & out["long_zone"].shift(1, fill_value=False)
        out["short_confirmed"] = out["short_zone"] & out["short_zone"].shift(1, fill_value=False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        ranks = indicators["rank"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        longs = indicators["long_confirmed"].to_numpy(dtype=bool)
        shorts = indicators["short_confirmed"].to_numpy(dtype=bool)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        armed = 0
        arm_index = -1
        entry_index = -1
        entry_atr = np.nan
        stop = np.nan
        for i in range(len(data)):
            if position:
                if i == entry_index:
                    # Read the entry bar's open only when that bar has arrived.
                    stop = opens[i] - position * params.stop_atr * entry_atr
                invalid = not np.isfinite(closes[i]) or not np.isfinite(stop)
                stopped = position * (closes[i] - stop) < 0
                expired = i - entry_index + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    position = 0
                    armed = 0
                targets[i] = position
                continue
            if not (np.isfinite(ranks[i]) and np.isfinite(atrs[i]) and atrs[i] > 0 and np.isfinite(closes[i])):
                armed = 0
                continue
            if armed and i - arm_index > params.arm_bars:
                armed = 0
            if ranks[i] <= params.tail_rank:
                armed, arm_index = 1, i
            elif ranks[i] >= 1.0 - params.tail_rank:
                armed, arm_index = -1, i
            if (armed == 1 and longs[i]) or (armed == -1 and shorts[i]):
                position = armed
                entry_index = i + 1
                entry_atr = atrs[i]
                armed = 0
            targets[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = targets
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = StrategyParams(stop_atr=2.0)
    index = pd.date_range("2020-01-01", periods=16, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 111.0, "low": 89.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    data.iloc[[3, 5, 10, 12], data.columns.get_loc("close")] = [110.0, 97.0, 90.0, 103.0]
    rank = pd.Series([0.1, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.9, 0.4, 0.3] + [0.5] * 6, index=index)
    ind = pd.DataFrame({"rank": rank, "atr": 1.0}, index=index)
    for name, zone in [("long", rank.gt(0.55)), ("short", rank.lt(0.45))]:
        ind[name + "_confirmed"] = zone & zone.shift(1, fill_value=False)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    expected = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0]
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == expected
    data["close"] = 100.0
    params.hold_bars = 3
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == expected
    assert strategy.indicators(data, params)["frac_close"].isna().all()


if __name__ == "__main__":
    _self_check()
