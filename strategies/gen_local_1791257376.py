from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    consistency_bars: int = 3
    rank_window: int = 60
    rank_threshold: float = 0.8
    atr_window: int = 14
    breakeven_pct: float = 0.008
    initial_atr: float = 2.0
    trail_atr: float = 1.6
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791257376"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.consistency_bars + params.rank_window + 1,
                       params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["floor"] = out["return"].rolling(params.consistency_bars).min()
        out["ceiling"] = out["return"].rolling(params.consistency_bars).max()
        out["floor_rank"] = out["floor"].rolling(params.rank_window).rank(pct=True)
        out["ceiling_rank"] = (-out["ceiling"]).rolling(params.rank_window).rank(pct=True)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["ma200"] = data["close"].rolling(200).mean()
        long_entry = (
            (data["close"] > out["ma200"]) & (out["floor"] > 0)
            & (out["floor_rank"] >= params.rank_threshold)
            & (out["floor_rank"].shift(1) < params.rank_threshold)
        )
        short_entry = (
            (data["close"] < out["ma200"]) & (out["ceiling"] < 0)
            & (out["ceiling_rank"] >= params.rank_threshold)
            & (out["ceiling_rank"].shift(1) < params.rank_threshold)
        )
        ready = np.isfinite(out[["floor_rank", "ceiling_rank", "atr", "ma200"]]).all(axis=1)
        ready &= out["atr"] > 0
        out["entry"] = np.select(
            [long_entry & ready, short_entry & ready], [1, -1], default=0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry = np.nan
        stop = peak = entry_atr = 0.0
        age = 0
        armed = False
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].to_numpy(dtype=int)
        for i, (opening, high, low, close) in enumerate(prices):
            if position:
                if not np.isfinite([opening, high, low, close, atr[i]]).all() or atr[i] <= 0:
                    position = 0
                    continue
                if np.isnan(entry):
                    # Initialize on the fill bar, never from a future opening.
                    entry = opening
                    peak = position * entry
                    stop = peak - params.initial_atr * entry_atr
                age += 1
                # Signed prices make the stop ratchet upward for both directions.
                favorable = high if position == 1 else -low
                adverse = low if position == 1 else -high
                hit = adverse <= stop
                peak = max(peak, favorable)
                armed = armed or peak - position * entry >= entry * params.breakeven_pct
                if armed:
                    stop = max(stop, position * entry, peak - params.trail_atr * atr[i])
                # Test old stop intrabar; a newly raised stop applies at the close.
                # Exit decisions execute at the next open through the signal shift.
                if hit or position * close <= stop or age >= params.hold_bars:
                    position = 0
                    continue
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(entries[i])
                entry = np.nan
                entry_atr = atr[i]
                age = 0
                armed = False
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params(hold_bars=5)
    for direction in (1, -1):
        index = pd.date_range("2020-01-01", periods=208)
        close = np.full(208, 100.0)
        close[202:205] = 100.0 + direction * np.array([1.0, 2.0, 0.0])
        data = pd.DataFrame({
            "open": close, "high": close + 0.25, "low": close - 0.25,
            "close": close, "volume": 1000.0,
        }, index=index)
        derived = strategy.indicators(data, params)
        assert derived.index.equals(data.index)
        derived["atr"] = 1.0
        derived["entry"] = 0
        derived.loc[index[200], "entry"] = direction
        ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
        result = strategy.generate_signals(data, derived, ctx, params).data
        assert result["signal"].iloc[200:206].tolist() == [0, direction, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
