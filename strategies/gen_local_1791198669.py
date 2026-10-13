from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 16
    remote_min: int = 3
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791198669"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params):
        return int(max(200, params.graph_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        if not (params.graph_window >= 6 and 1 <= params.remote_min <= params.graph_window - 4
                and params.atr_window >= 1 and params.stop_atr > 0
                and 5 <= params.hold_bars <= 10):
            raise ValueError("Invalid visibility, ATR, stop, or holding parameters")
        close = data["close"]
        spread = (data["high"] - data["low"]).where(lambda x: x > 0)
        pressure = ((close - data["open"]) / spread).replace([np.inf, -np.inf], np.nan)
        degree = np.zeros(len(data), dtype=int)
        remote = np.zeros(len(data), dtype=int)
        max_slope = np.full(len(data), -np.inf)
        # Natural visibility: every intermediate vertex lies strictly below the edge.
        for lag in range(1, params.graph_window + 1):
            slope = ((pressure.shift(lag) - pressure) / lag).to_numpy()
            finite = np.isfinite(slope)
            visible = finite & (slope > max_slope)
            degree += visible
            if lag >= 5:
                remote += visible
            max_slope = np.where(finite, np.maximum(max_slope, slope), np.inf)
        complete = pressure.rolling(params.graph_window + 1).count().eq(params.graph_window + 1)
        previous = close.shift(1)
        tr = pd.concat([(data["high"] - data["low"]),
                        (data["high"] - previous).abs(),
                        (data["low"] - previous).abs()], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["pressure"] = pressure
        out["degree"] = pd.Series(degree, index=data.index).where(complete)
        out["remote_degree"] = pd.Series(remote, index=data.index).where(complete)
        out["remote_gain"] = out["remote_degree"].diff()
        out["atr"] = tr.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["turn_month"] = (data.index.day >= 25) | (data.index.day <= 3)
        out["ready"] = np.isfinite(out[["pressure", "remote_degree", "remote_gain", "atr", "ma200"]]).all(axis=1)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        eligible = (indicators["ready"] & indicators["turn_month"]
                    & (data["close"] > indicators["ma200"])
                    & (indicators["pressure"] > 0)
                    & (indicators["remote_degree"] >= params.remote_min)
                    & (indicators["remote_gain"] > 0)
                    & (indicators["atr"] > 0)).fillna(False).to_numpy()
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        stop = np.nan
        entered = -1
        for i in range(len(data)):
            if pending:
                # Previous close scheduled this fill; freeze the entry-open ATR stop.
                active = bool(np.isfinite(opens[i]) and opens[i] > 0)
                stop = opens[i] - params.stop_atr * atr[i - 1]
                entered = i
                pending = False
            if active:
                if (not np.isfinite(close[i]) or not np.isfinite(ma[i])
                        or close[i] < stop or close[i] <= ma[i]
                        or i - entered + 1 >= params.hold_bars):
                    active = False
                else:
                    raw[i] = 1
            elif eligible[i]:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    p = VisibilityParams()
    s = GeneratedStrategy()
    ix = pd.date_range("2020-01-01", periods=12, freq="B")
    d = pd.DataFrame({"open": 100.0, "high": 106.0, "low": 94.0,
                      "close": 100.0, "volume": 1000.0}, index=ix)
    ind = pd.DataFrame({"ready": True, "turn_month": False, "ma200": 90.0,
                        "pressure": 0.5, "remote_degree": 3.0,
                        "remote_gain": 1.0, "atr": 2.0}, index=ix)
    ind.loc[ix[0], "turn_month"] = True
    d.loc[ix[2:5], "close"] = [105.0, 97.0, 95.0]
    result = s.generate_signals(d, ind, None, p).data
    assert result["signal"].tolist()[:6] == [0, 1, 1, 1, 1, 0]
    d["close"] = 100.0
    result = s.generate_signals(d, ind, None, p).data
    assert result["signal"].sum() == p.hold_bars
    flat = pd.concat([d] * 25, ignore_index=True)
    flat.index = pd.date_range("2020-01-01", periods=len(flat), freq="B")
    topology = s.indicators(flat, p)
    assert topology["remote_degree"].dropna().eq(0).all()
    prefix = s.indicators(flat.iloc[:240], p)
    assert topology.iloc[:240].equals(prefix)
