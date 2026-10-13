from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologySeasonalityParams:
    graph_bars: int = 20
    atr_bars: int = 14
    history_bars: int = 504
    max_hold: int = 8
    min_samples: int = 6
    topology_threshold: float = 0.25
    min_edge: float = 0.001
    breakeven_pct: float = 0.01
    initial_stop_atr: float = 2.5
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[TopologySeasonalityParams]):
    strategy_id = "gen_local-w2_1791336848"

    @classmethod
    def params_type(cls):
        return TopologySeasonalityParams

    def warmup_bars(self, params: TopologySeasonalityParams) -> int:
        return int(max(params.graph_bars + params.max_hold + params.history_bars,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: TopologySeasonalityParams) -> pd.DataFrame:
        if (params.graph_bars < 3 or params.atr_bars < 1 or params.history_bars < 1
                or params.max_hold < 1 or not 1 <= params.min_samples <= params.history_bars
                or not 0 < params.topology_threshold <= 1 or params.min_edge < 0
                or params.breakeven_pct <= 0 or params.initial_stop_atr <= 0
                or params.trail_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(data["close"].where(data["close"] > 0))
        price = out["log_close"].to_numpy(dtype=float)
        upper = np.zeros(len(data))
        lower = np.zeros(len(data))
        highest = np.full(len(data), -np.inf)
        lowest = np.full(len(data), np.inf)
        # Endpoint visibility: an edge exists when its slope clears all nearer slopes.
        for lag in range(1, params.graph_bars):
            older = out["log_close"].shift(lag).to_numpy(dtype=float)
            slope = (older - price) / lag
            valid = np.isfinite(slope)
            upper += valid & (slope > highest)
            lower += valid & (slope < lowest)
            highest = np.maximum(highest, np.where(valid, slope, -np.inf))
            lowest = np.minimum(lowest, np.where(valid, slope, np.inf))
        full = out["log_close"].rolling(params.graph_bars).count().eq(params.graph_bars)
        out["upper_degree"] = pd.Series(upper, index=data.index).where(full)
        out["lower_degree"] = pd.Series(lower, index=data.index).where(full)
        out["topology"] = ((out["upper_degree"] - out["lower_degree"])
                           / (out["upper_degree"] + out["lower_degree"]).replace(0, np.nan))
        previous = data["close"].shift(1)
        ranges = pd.concat([data["high"] - data["low"],
                            (data["high"] - previous).abs(),
                            (data["low"] - previous).abs()], axis=1)
        out["atr"] = ranges.max(axis=1).rolling(params.atr_bars).mean()
        # Labels appear only after the complete holding horizon has elapsed.
        out["matured_return"] = data["close"] / data["close"].shift(params.max_hold) - 1.0
        old_topology = out["topology"].shift(params.max_hold)
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        old_weekday = weekday.shift(params.max_hold)
        out["edge"] = np.nan
        for day in range(5):
            for polarity in (-1, 1):
                cohort = old_weekday.eq(day) & (old_topology * polarity >= params.topology_threshold)
                sample = out["matured_return"].where(cohort).replace([np.inf, -np.inf], np.nan)
                estimate = sample.rolling(params.history_bars, min_periods=params.min_samples).mean()
                current = weekday.eq(day) & (out["topology"] * polarity >= params.topology_threshold)
                out["edge"] = out["edge"].where(~current, estimate)
        out["ready"] = (out[["topology", "atr", "edge"]].notna().all(axis=1)
                        & out["atr"].gt(0)
                        & (np.arange(len(data)) >= self.warmup_bars(params) - 1))
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: TopologySeasonalityParams) -> SignalFrame:
        eligible = indicators["ready"].fillna(False) & indicators["edge"].abs().ge(params.min_edge)
        candidate = pd.Series(np.where(eligible, np.sign(indicators["edge"]), 0),
                              index=data.index).fillna(0).astype(int)
        confirmed = candidate.ne(0) & candidate.eq(candidate.shift(1).fillna(0))
        raw = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = candidate.where(confirmed, 0).to_numpy(dtype=int)
        side, age = 0, 0
        entry, stop = np.nan, np.nan
        armed = False
        for i in range(len(data)):
            if side:
                if not np.isfinite(entry):
                    # This open is observed only after the previous close's entry decision.
                    entry = opens[i]
                    if (not np.isfinite(entry) or entry <= 0 or i == 0
                            or not np.isfinite(atrs[i - 1]) or atrs[i - 1] <= 0):
                        side = 0
                        continue
                    stop = side * entry - params.initial_stop_atr * atrs[i - 1]
                age += 1
                close = closes[i]
                if not np.isfinite(close) or close <= 0:
                    side = 0
                    continue
                signed_close = side * close
                # Close-observed exits execute next open after the mandatory signal shift.
                if signed_close <= stop or age >= params.max_hold:
                    side = 0
                    entry = np.nan
                    continue
                if side * (close / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    # Ratchet upward in signed prices; short buy-stop prices ratchet downward.
                    stop = max(stop, side * entry)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, signed_close - params.trail_atr * atrs[i])
                raw[i] = side
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                entry, stop = np.nan, np.nan
                age, armed = 0, False
                raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = TopologySeasonalityParams()
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0.0, 0.0, 1.5, 3.0, 0.25, 0.0])
        data = pd.DataFrame({"open": 100.0, "high": close + 1.0,
                             "low": close - 1.0, "close": close, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"atr": 1.0, "edge": direction * 0.01, "ready": True}, index=index)
        signals = strategy.generate_signals(data, ind, ctx, params).data
        assert signals["signal"].tolist() == [0, 0, direction, direction, direction, 0]
        assert signals["size"].gt(0).all()
        isolated = ind.copy()
        isolated["edge"] = [0.01, 0.0, -0.01, 0.0, 0.01, 0.0]
        assert strategy.generate_signals(data, isolated, ctx, params).data["signal"].eq(0).all()
