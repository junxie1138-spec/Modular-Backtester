from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapAnalogParams:
    history_bars: int = 504
    shape_bars: int = 3
    scale_bars: int = 20
    neighbors: int = 12
    max_distance: float = 2.0
    min_edge: float = 0.002
    min_win_rate: float = 0.60
    profit_target: float = 0.018
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GapAnalogParams]):
    strategy_id = "gen_local_1791327209"

    @classmethod
    def params_type(cls):
        return GapAnalogParams

    @staticmethod
    def warmup_bars(params: GapAnalogParams) -> int:
        return int(max(200, params.scale_bars + params.shape_bars
                       + params.history_bars + params.hold_bars + 1))

    def indicators(self, data: pd.DataFrame, params: GapAnalogParams) -> pd.DataFrame:
        if (params.history_bars < params.neighbors or params.neighbors < 1
                or params.shape_bars < 1 or params.scale_bars < 2
                or not 3 <= params.hold_bars <= 5
                or params.profit_target <= 0 or params.max_distance <= 0
                or not 0 <= params.min_win_rate <= 1):
            raise ValueError("Invalid gap-analog parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["gap"] = opening.div(close.shift(1)).sub(1)
        out["gap_scale"] = np.sqrt(
            out["gap"].pow(2).shift(1).rolling(
                params.scale_bars, min_periods=params.scale_bars).mean()
        ).clip(lower=0.001)
        out["normalized_gap"] = out["gap"].div(out["gap_scale"])
        feature_columns = []
        for lag in range(params.shape_bars):
            name = f"gap_lag_{lag}"
            out[name] = out["normalized_gap"].shift(lag)
            feature_columns.append(name)
        out["body"] = close.div(opening).sub(1)
        out["normalized_body"] = out["body"].div(out["gap_scale"])
        feature_columns.append("normalized_body")
        out["bull_regime"] = close.gt(out["ma200"])
        # At t, this labels the setup at t-h using its next open and h-th close.
        out["matured_return"] = close.div(
            opening.shift(params.hold_bars - 1)).sub(1)
        features = out[feature_columns].to_numpy(dtype=float)
        outcomes = out["matured_return"].to_numpy(dtype=float)
        regime = out["bull_regime"].to_numpy(dtype=bool)
        setup = (out["gap"].lt(0) & out["body"].gt(0)).to_numpy(dtype=bool)
        edge = np.full(len(data), np.nan)
        win_rate = np.full(len(data), np.nan)
        distance = np.full(len(data), np.nan)
        count = np.zeros(len(data), dtype=int)
        # ponytail: bounded O(bars * history) search; index neighbors if history grows.
        for i in range(self.warmup_bars(params), len(data)):
            if not (regime[i] and setup[i] and np.isfinite(features[i]).all()):
                continue
            # One-bar embargo after each historical outcome has matured.
            end = i - params.hold_bars
            start = max(0, end - params.history_bars)
            candidates = np.arange(start, end)
            valid = (regime[candidates] & setup[candidates]
                     & np.isfinite(features[candidates]).all(axis=1)
                     & np.isfinite(outcomes[candidates + params.hold_bars]))
            candidates = candidates[valid]
            if len(candidates) < params.neighbors:
                continue
            distances = np.mean((features[candidates] - features[i]) ** 2, axis=1)
            nearest = np.argsort(distances)[:params.neighbors]
            if distances[nearest[-1]] > params.max_distance ** 2:
                continue
            returns = outcomes[candidates[nearest] + params.hold_bars]
            edge[i] = float(np.median(returns))
            win_rate[i] = float(np.mean(returns > 0))
            distance[i] = float(np.sqrt(distances[nearest[-1]]))
            count[i] = params.neighbors
        out["analog_edge"] = edge
        out["analog_win_rate"] = win_rate
        out["analog_distance"] = distance
        out["analog_count"] = count
        out["entry"] = (out["bull_regime"] & out["gap"].lt(0)
                        & out["body"].gt(0) & out["analog_edge"].ge(params.min_edge)
                        & out["analog_win_rate"].ge(params.min_win_rate)).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GapAnalogParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opening = data["open"].to_numpy(dtype=float)
        close = data["close"].to_numpy(dtype=float)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opening[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    entry_bar = -1
                    continue
                target_hit = (np.isfinite(close[i])
                              and close[i] >= entry_price * (1 + params.profit_target))
                expired = i - entry_bar + 1 >= params.hold_bars
                if target_hit or expired:
                    entry_bar = -1
                else:
                    raw[i] = 1
            elif entries[i] and i >= self.warmup_bars(params):
                raw[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapAnalogParams()
    strategy = GeneratedStrategy()
    t = strategy.warmup_bars(params)
    index = pd.date_range("2000-01-01", periods=t + 10, freq="D")
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index)
    entries = pd.DataFrame({"entry": False}, index=index)
    entries.iloc[t, 0] = True
    signal = strategy.generate_signals(data, entries, None, params).data["signal"]
    assert signal.iloc[t] == 0 and signal.iloc[t + 1:t + 5].eq(1).all()
    assert signal.iloc[t + 5] == 0
    data.iloc[t + 2, data.columns.get_loc("close")] = 103.0
    signal = strategy.generate_signals(data, entries, None, params).data["signal"]
    assert signal.iloc[t + 1:t + 3].eq(1).all() and signal.iloc[t + 3] == 0
    computed = strategy.indicators(data, params)
    cutoff = t + 3
    pd.testing.assert_frame_equal(
        computed.iloc[:cutoff], strategy.indicators(data.iloc[:cutoff], params))
    assert computed.index.equals(data.index)


if __name__ == "__main__":
    _self_check()
