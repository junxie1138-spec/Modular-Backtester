from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DrawdownQueueParams:
    peak_window: int = 28
    load_window: int = 7
    rank_window: int = 63
    capacity_quantile: float = 0.65
    vol_window: int = 10
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[DrawdownQueueParams]):
    strategy_id = "gen_local_1791272187"

    @classmethod
    def params_type(cls):
        return DrawdownQueueParams

    def warmup_bars(self, params):
        return int(max(params.peak_window + params.load_window + params.rank_window + 1, params.vol_window + 1))

    def indicators(self, data, params):
        if min(params.peak_window, params.load_window, params.rank_window, params.vol_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not 0.0 < params.capacity_quantile < 1.0 or not np.isfinite(params.target_vol) or params.target_vol <= 0.0:
            raise ValueError("Invalid percentile or volatility target.")
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["close"].rolling(params.peak_window).max()
        out["depth"] = (1.0 - data["close"] / out["peak"].replace(0.0, np.nan)).clip(lower=0.0)
        out["load"] = out["depth"].rolling(params.load_window).mean()
        out["flow"] = out["load"].diff()
        out["capacity"] = out["load"].rolling(params.rank_window).quantile(params.capacity_quantile).shift(1)
        out["flow_median"] = out["flow"].rolling(params.rank_window).quantile(0.5).shift(1)
        out["returns"] = data["close"].pct_change(fill_method=None)
        out["vol"] = out["returns"].rolling(params.vol_window).std(ddof=0) * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        required = ["load", "flow", "capacity", "flow_median", "vol"]
        ready = indicators[required].notna().all(axis=1)
        saturated = indicators["load"] > indicators["capacity"]
        draining = indicators["flow"] < indicators["flow_median"].clip(upper=0.0)
        filling = indicators["flow"] > indicators["flow_median"].clip(lower=0.0)
        entries = np.select([ready & saturated & draining, ready & saturated & filling], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        size = (params.target_vol / indicators["vol"].replace(0.0, np.nan)).clip(lower=0.01, upper=0.95)
        df["size"] = size.shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.01).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = DrawdownQueueParams()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": np.ones(6)}, index=index)
    indicators = pd.DataFrame({
        "load": np.full(6, 2.0),
        "capacity": np.ones(6),
        "flow": [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        "flow_median": np.zeros(6),
        "vol": np.full(6, 0.2),
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert np.isfinite(result["size"]).all() and (result["size"] > 0.0).all()
