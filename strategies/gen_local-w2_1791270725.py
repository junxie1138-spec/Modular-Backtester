from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 63
    complexity_gap: float = 0.15
    seasonal_floor: float = 0.0002
    stop_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791270725"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(631, params.pattern_window + 2, 15)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.pattern_window < 6 or params.hold_bars < 1:
            raise ValueError("pattern_window must be >= 6 and hold_bars >= 1")
        if params.stop_atr <= 0 or params.complexity_gap < 0 or params.seasonal_floor < 0:
            raise ValueError("Invalid thresholds")
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        code = ((a < b).astype(int) + 2 * (a < c).astype(int)
                + 4 * (b < c).astype(int)).where(valid)
        out = pd.DataFrame(index=data.index)
        out["ordinal_code"] = code
        for motif in (0, 1, 3, 4, 6, 7):
            out[f"p{motif}"] = code.eq(motif).astype(float).where(code.notna()).rolling(
                params.pattern_window, min_periods=params.pattern_window
            ).mean()
        up = out[["p3", "p6", "p7"]].sum(axis=1, min_count=3)
        down = out[["p0", "p1", "p4"]].sum(axis=1, min_count=3)
        out["up_mass"], out["down_mass"] = up, down
        for label, motifs, mass in (("up", (3, 6, 7), up), ("down", (0, 1, 4), down)):
            q = out[[f"p{x}" for x in motifs]].div(mass.replace(0, np.nan), axis=0)
            terms = q * np.log(q.where(q > 0, 1.0))
            out[f"{label}_entropy"] = -terms.sum(axis=1, min_count=3) / np.log(3.0)
        out["complexity_edge"] = out["down_entropy"] - out["up_entropy"]
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return"] = returns
        season = pd.Series(np.nan, index=data.index, dtype=float)
        # ponytail: 126 bars approximates six months; use calendar offsets if drift matters.
        for month in range(1, 13):
            samples = returns.where(data.index.month == month).shift(126)
            count = samples.notna().astype(float).rolling(504, min_periods=504).sum()
            total = samples.fillna(0.0).rolling(504, min_periods=504).sum()
            estimate = total.div(count.where(count >= 20))
            season = season.where(data.index.month != month, estimate)
        out["seasonal_drift"] = season
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        edge, season = indicators["complexity_edge"], indicators["seasonal_drift"]
        long = ((edge > params.complexity_gap) & (indicators["up_mass"] >= 0.5)
                & (season > params.seasonal_floor))
        short = ((edge < -params.complexity_gap) & (indicators["down_mass"] >= 0.5)
                 & (season < -params.seasonal_floor))
        ready = (np.isfinite(indicators["atr"]) & indicators["atr"].gt(0)
                 & np.isfinite(edge) & np.isfinite(season))
        setup = np.select([long & ready, short & ready], [1, -1], default=0)
        targets = np.zeros(len(data), dtype=int)
        position, entry_bar = 0, -1
        entry, distance = np.nan, np.nan
        opens, closes = data["open"].to_numpy(), data["close"].to_numpy()
        atr = indicators["atr"].to_numpy()
        for i in range(len(data)):
            if position:
                if np.isnan(entry):
                    entry, distance = opens[i], params.stop_atr * atr[i - 1]
                    entry_bar = i
                invalid = not (np.isfinite(entry) and np.isfinite(closes[i])
                               and np.isfinite(distance) and distance > 0)
                stopped = (position * (closes[i] - entry) < -distance)
                expired = i - entry_bar + 1 >= params.hold_bars
                if invalid or stopped or expired:
                    position, entry = 0, np.nan
                targets[i] = position
            elif setup[i] and np.isfinite(closes[i]):
                position, entry = int(setup[i]), np.nan
                targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(stop_atr=2.0, hold_bars=18)
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 111.0, "low": 89.0,
                         "close": [100., 102., 110., 105., 95., 100., 100.],
                         "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"complexity_edge": [0.5, 0., 0., 0., 0., 0., 0.],
                        "seasonal_drift": 0.001, "up_mass": 0.7,
                        "down_mass": 0.3, "atr": 2.0}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    ind["seasonal_drift"] = -0.001
    assert not strategy.generate_signals(data, ind, None, params).data["signal"].any()
    ind["complexity_edge"] = [-0.5, 0., 0., 0., 0., 0., 0.]
    ind["up_mass"], ind["down_mass"] = 0.3, 0.7
    data["close"] = [100., 98., 90., 105., 100., 100., 100.]
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, -1, 0, 0, 0]
    params.hold_bars = 2
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0]


if __name__ == "__main__":
    _self_check()
