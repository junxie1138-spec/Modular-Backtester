from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityGapParams:
    graph_window: int = 20
    rank_window: int = 126
    atr_window: int = 14
    topology_percentile: float = 0.80
    gap_percentile: float = 0.20
    breakeven_pct: float = 0.01
    trail_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[VisibilityGapParams]):
    strategy_id = "gen_local_1791215094"

    @classmethod
    def params_type(cls):
        return VisibilityGapParams

    @staticmethod
    def warmup_bars(params):
        return int(params.atr_window + params.graph_window + params.rank_window + 2)

    @staticmethod
    def indicators(data, params):
        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        gap = (data["open"] / previous_close.where(previous_close > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        prior_atr_fraction = atr.shift(1) / previous_close.where(previous_close > 0)
        shock = (-gap / prior_atr_fraction.where(prior_atr_fraction > 0)).replace([np.inf, -np.inf], np.nan)

        # Horizontal visibility: both endpoints must exceed every intervening node.
        degree = pd.Series(0.0, index=data.index)
        blockers = pd.Series(-np.inf, index=data.index)
        for distance in range(1, params.graph_window + 1):
            past = shock.shift(distance)
            visible = shock.notna() & past.notna() & (shock > blockers) & (past > blockers)
            degree += visible.astype(float)
            blockers = pd.concat([blockers, past], axis=1).max(axis=1)
        complete = shock.rolling(params.graph_window + 1, min_periods=params.graph_window + 1).count().eq(params.graph_window + 1)
        degree = degree.where(complete)
        topology_threshold = degree.shift(1).rolling(params.rank_window, min_periods=params.rank_window).quantile(params.topology_percentile)
        gap_threshold = gap.shift(1).rolling(params.rank_window, min_periods=params.rank_window).quantile(params.gap_percentile)
        recovery_level = (data["open"] + previous_close) / 2.0
        entry = (
            (gap < 0)
            & (gap <= gap_threshold)
            & (degree >= topology_threshold)
            & (data["close"] > data["open"])
            & (data["close"] >= recovery_level)
            & atr.gt(0)
            & topology_threshold.notna()
            & gap_threshold.notna()
        ).fillna(False)
        return pd.DataFrame({
            "previous_close": previous_close,
            "true_range": true_range,
            "atr": atr,
            "gap": gap,
            "prior_atr_fraction": prior_atr_fraction,
            "shock": shock,
            "degree": degree,
            "topology_threshold": topology_threshold,
            "gap_threshold": gap_threshold,
            "recovery_level": recovery_level,
            "entry": entry,
        }, index=data.index)

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        raw = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        high_water = 0.0
        age = 0

        for i in range(len(data)):
            valid = (
                np.isfinite(opens[i]) and opens[i] > 0
                and np.isfinite(highs[i]) and highs[i] > 0
                and np.isfinite(closes[i]) and closes[i] > 0
                and np.isfinite(atrs[i]) and atrs[i] > 0
            )
            if not valid:
                pending = active = False
                continue
            if pending:
                pending = False
                if i == 0 or not np.isfinite(atrs[i - 1]) or atrs[i - 1] <= 0:
                    continue
                # Yesterday's decision fills at today's open.
                entry_price = opens[i]
                stop = entry_price - params.trail_atr * atrs[i - 1]
                high_water = entry_price
                armed = False
                age = 0
                active = True
            if active:
                age += 1
                high_water = max(high_water, highs[i])
                if high_water >= entry_price * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry_price, high_water - params.trail_atr * atrs[i])
                # Stops are evaluated at the close; liquidation fills next open.
                if closes[i] <= stop or age >= params.hold_bars:
                    active = False
                    continue
                raw[i] = 1
            elif entries[i] and i + 1 < len(data):
                raw[i] = 1
                pending = True

        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"atr": 1.0, "entry": [True, False, False, False, False, False, False]}, index=index)
    params = VisibilityGapParams()
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data.loc[index[1], ["high", "close"]] = [102.0, 101.0]
    data.loc[index[2], "close"] = 99.0
    result = GeneratedStrategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= params.graph_window + params.rank_window + params.atr_window + 1


if __name__ == "__main__":
    _self_check()
