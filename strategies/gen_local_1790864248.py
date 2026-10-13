from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class PlateauParams:
    peak_window: int = 63
    depth_window: int = 20
    plateau_window: int = 5
    vol_window: int = 20
    min_depth: float = 0.025
    max_depth: float = 0.12
    compression: float = 0.30
    target_vol: float = 0.12
    profit_target: float = 0.04
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[PlateauParams]):
    strategy_id = "gen_local_1790864248"

    @classmethod
    def params_type(cls):
        return PlateauParams

    def warmup_bars(self, params: PlateauParams) -> int:
        return int(max(
            params.peak_window + max(params.depth_window, params.plateau_window) - 1,
            params.vol_window + 1,
        ))

    def indicators(self, data: pd.DataFrame, params: PlateauParams) -> pd.DataFrame:
        p = params
        windows = (p.peak_window, p.depth_window, p.plateau_window, p.vol_window)
        values = (p.min_depth, p.max_depth, p.compression, p.target_vol, p.profit_target)
        if (any(w < 2 for w in windows) or p.plateau_window > p.depth_window
                or p.hold_bars < 1 or not all(np.isfinite(v) for v in values)
                or not 0 < p.min_depth < p.max_depth < 1
                or not 0 < p.compression < 1 or p.target_vol <= 0
                or p.profit_target <= 0):
            raise ValueError("Invalid plateau parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["returns"] = close.pct_change(fill_method=None)
        out["peak"] = close.rolling(p.peak_window, min_periods=p.peak_window).max()
        out["depth"] = (1.0 - close / out["peak"]).clip(lower=0.0)
        out["recent_high"] = out["depth"].rolling(p.plateau_window).max()
        out["recent_low"] = out["depth"].rolling(p.plateau_window).min()
        out["broad_high"] = out["depth"].rolling(p.depth_window).max()
        out["broad_low"] = out["depth"].rolling(p.depth_window).min()
        out["recent_range"] = out["recent_high"] - out["recent_low"]
        out["broad_range"] = out["broad_high"] - out["broad_low"]
        out["range_ratio"] = out["recent_range"] / out["broad_range"].where(out["broad_range"] > 0)
        out["stability"] = (1.0 - out["range_ratio"]).clip(0.0, 1.0)
        out["annual_vol"] = out["returns"].rolling(p.vol_window).std(ddof=0) * np.sqrt(252.0)
        out["strength"] = (out["depth"] / (2.0 * p.min_depth)).clip(0.0, 1.0) * out["stability"]
        ready = np.isfinite(out).all(axis=1) & (out["annual_vol"] > 0)
        eligible = (
            ready
            & out["depth"].between(p.min_depth, p.max_depth)
            & (out["depth"] >= 0.85 * out["broad_high"])
            & (out["range_ratio"] <= p.compression)
        )
        out["entry"] = eligible & ~eligible.shift(1, fill_value=False)
        out["allocation"] = (
            p.target_vol / out["annual_vol"].where(out["annual_vol"] > 0)
            * out["strength"]
        ).clip(0.05, 1.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: PlateauParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        entry_bar = -1
        entry_price = np.nan
        allocation = 1.0
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        triggers = indicators["entry"].to_numpy(dtype=bool)
        weights = indicators["allocation"].to_numpy(dtype=float)
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    entry_price = opens[i]
                price_valid = np.isfinite(entry_price) and entry_price > 0
                target_hit = price_valid and closes[i] >= entry_price * (1.0 + params.profit_target)
                timed_out = i - entry_bar + 1 >= params.hold_bars
                if not price_valid or target_hit or timed_out:
                    entry_bar = -1
                    continue
                signals[i] = 1
                sizes[i] = allocation
            elif triggers[i] and np.isfinite(closes[i]) and closes[i] > 0:
                weight = weights[i]
                if not np.isfinite(weight) or weight <= 0:
                    continue
                entry_bar = i + 1
                allocation = float(np.clip(weight, 0.05, 1.0))
                signals[i] = 1
                sizes[i] = allocation
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = PlateauParams(hold_bars=3, profit_target=0.03)
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 105.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"entry": [True] + [False] * 7, "allocation": 0.4}, index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert np.allclose(result.loc[index[1:4], "size"], 0.4)
    data.loc[index[1], "close"] = 104.0
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert strategy.indicators(data, PlateauParams())["entry"].sum() == 0
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
