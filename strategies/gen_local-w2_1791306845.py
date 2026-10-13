from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 63
    coupling_window: int = 40
    percentile_window: int = 126
    depth_percentile: float = 0.80
    coupling_percentile: float = 0.25
    profit_target: float = 0.015
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791306845"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return params.peak_window + params.coupling_window + params.percentile_window + 2

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].where(data["close"] > 0)
        peak = close.rolling(params.peak_window, min_periods=params.peak_window).max()
        depth = 1.0 - close / peak
        change = depth.diff()
        shock = change.clip(lower=0).shift(1)
        coupling = shock.rolling(params.coupling_window, min_periods=params.coupling_window).corr(change)
        coupling = coupling.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        depth_threshold = depth.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.depth_percentile)
        coupling_threshold = coupling.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.coupling_percentile)
        return pd.DataFrame({
            "peak": peak,
            "depth": depth,
            "depth_change": change,
            "lagged_downside_shock": shock,
            "coupling": coupling,
            "depth_threshold": depth_threshold,
            "coupling_threshold": coupling_threshold,
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        ready = indicators.notna().all(axis=1)
        setup = (
            ready
            & (indicators["depth"] >= indicators["depth_threshold"])
            & (indicators["depth_change"] > 0)
            & (indicators["coupling"] < 0)
            & (indicators["coupling"] <= indicators["coupling_threshold"])
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                # Yesterday's entry decision fills at today's open.
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        entry_bar = -1
                        continue
                reached_target = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                timed_out = i - entry_bar + 1 >= params.max_hold
                if reached_target or timed_out:
                    entry_bar = -1
                else:
                    signal[i] = 1
            elif setup[i] and np.isfinite(closes[i]) and closes[i] > 0:
                signal[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({
        "depth": [0.1, 0, 0, 0, 0, 0],
        "depth_change": 0.01,
        "coupling": -0.5,
        "depth_threshold": 0.05,
        "coupling_threshold": -0.2,
    }, index=index)
    strategy = GeneratedStrategy()
    params = Params(max_hold=3, profit_target=0.02)
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 103.0
    assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [0, 1, 0, 0, 0, 0]
