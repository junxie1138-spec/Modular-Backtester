from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 20
    memory_window: int = 63
    strength_window: int = 63
    percentile: float = 0.8
    min_occurrences: int = 5
    atr_window: int = 14
    breakeven_pct: float = 0.008
    trail_atr: float = 2.0
    hold_bars: int = 4

    def __post_init__(self):
        if min(self.rank_window, self.memory_window, self.strength_window, self.atr_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.5 < self.percentile < 1.0:
            raise ValueError("percentile must be between 0.5 and 1.")
        if not 1 <= self.min_occurrences <= self.memory_window:
            raise ValueError("Invalid minimum occurrence count.")
        if not 3 <= self.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five.")
        if not np.isfinite(self.breakeven_pct) or self.breakeven_pct <= 0:
            raise ValueError("breakeven_pct must be finite and positive.")
        if not np.isfinite(self.trail_atr) or self.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791214052"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(
            params.rank_window + params.memory_window + params.strength_window + 1,
            params.atr_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["rank"] = out["return"].rolling(params.rank_window).rank(pct=True)
        out["lag_rank"] = out["rank"].shift(1)
        ready = out["rank"].notna() & out["lag_rank"].notna()
        for tail, condition in (
            ("upper", out["lag_rank"] >= params.percentile),
            ("lower", out["lag_rank"] <= 1.0 - params.percentile),
        ):
            out[tail + "_event"] = condition.astype(float).where(ready)
            out[tail + "_response"] = (out["rank"] - 0.5) * out[tail + "_event"]
            out[tail + "_count"] = out[tail + "_event"].rolling(params.memory_window).sum()
            out[tail + "_memory"] = (
                out[tail + "_response"].rolling(params.memory_window).sum()
                / out[tail + "_count"].replace(0.0, np.nan)
            )
            out[tail + "_strength"] = out[tail + "_memory"].abs().rolling(params.strength_window).rank(pct=True)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1).where(data[["high", "low", "close"]].notna().all(axis=1))
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        desire = pd.Series(0, index=data.index, dtype=int)
        for tail, extreme in (
            ("upper", indicators["rank"] >= params.percentile),
            ("lower", indicators["rank"] <= 1.0 - params.percentile),
        ):
            eligible = (
                extreme
                & (indicators[tail + "_count"] >= params.min_occurrences)
                & (indicators[tail + "_strength"] >= params.percentile)
            )
            direction = np.sign(indicators[tail + "_memory"].fillna(0.0)).astype(int)
            desire.loc[eligible.fillna(False)] = direction.loc[eligible.fillna(False)]

        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        wishes = desire.to_numpy()
        raw = np.zeros(len(data), dtype=int)
        position, age, armed = 0, 0, False
        entry, stop, peak = np.nan, np.nan, np.nan
        for i in range(len(data)):
            opening, high, low, close = prices[i]
            if not np.isfinite(prices[i]).all() or min(prices[i]) <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                position, entry = 0, np.nan
                continue
            if position:
                if np.isnan(entry):
                    # The preceding close's decision fills at this session's open.
                    entry, age, armed = opening, 0, False
                    peak = position * entry
                    stop = peak - params.trail_atr * atr[i - 1]
                age += 1
                adverse = low if position == 1 else -high
                favorable = high if position == 1 else -low
                signed_close = position * close
                exit_now = adverse <= stop
                if not exit_now:
                    peak = max(peak, favorable)
                    armed = armed or peak - position * entry >= params.breakeven_pct * entry
                    if armed:
                        # Signed-price stop only rises; short buy-stop prices only fall.
                        stop = max(stop, position * entry, peak - params.trail_atr * atr[i])
                        exit_now = signed_close <= stop
                # Stop breaches schedule next-open exits, never same-bar fills.
                if exit_now or age >= params.hold_bars:
                    position, entry = 0, np.nan
                    continue
            elif wishes[i] != 0:
                position, entry = int(wishes[i]), np.nan
            raw[i] = position

        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=8)
    opening = np.array([100, 100, 101, 104, 101, 100, 97, 99], dtype=float)
    close = np.array([100, 101, 104, 101, 100, 97, 99, 100], dtype=float)
    data = pd.DataFrame({
        "open": opening, "high": np.maximum(opening, close) + 0.1,
        "low": np.minimum(opening, close) - 0.1,
        "close": close, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "rank": [0.9, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5],
        "upper_count": 10.0, "lower_count": 10.0,
        "upper_strength": 1.0, "lower_strength": 1.0,
        "upper_memory": 0.1, "lower_memory": -0.1, "atr": 1.0,
    }, index=index)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = [100.0, 100.1, 99.9, 100.0]
    ind["rank"] = [0.9] + [0.5] * 7
    result = strategy.generate_signals(flat, ind, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    warm = strategy.indicators(flat, params)
    assert strategy.generate_signals(flat, warm, ctx, params).data["signal"].eq(0).all()
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
