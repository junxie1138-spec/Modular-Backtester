from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    graph_window: int = 14
    topology_threshold: float = 0.20
    atr_window: int = 14
    stop_atr: float = 1.75
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791294536"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.graph_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.graph_window < 4 or params.atr_window < 1:
            raise ValueError("Lookbacks must be positive; graph_window must be at least four.")
        if not 3 <= params.hold_bars <= 5 or params.stop_atr <= 0:
            raise ValueError("hold_bars must be 3-5 and stop_atr must be positive.")
        if not 0 < params.topology_threshold < 1:
            raise ValueError("topology_threshold must be between zero and one.")
        close = data["close"].astype(float)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out = pd.DataFrame(index=data.index)
        out["returns"] = returns
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        width = params.graph_window
        windows = np.column_stack([returns.shift(lag).to_numpy() for lag in range(width - 1, -1, -1)])
        valid = np.isfinite(windows).all(axis=1)
        upper_degree = np.zeros_like(windows)
        lower_degree = np.zeros_like(windows)
        # Horizontal visibility: every intervening return must be below both endpoints.
        # ponytail: O(bars * window^2); chunk or compile if large graph windows become necessary.
        for left in range(width - 1):
            upper_barrier = np.full(len(data), -np.inf)
            lower_barrier = np.full(len(data), -np.inf)
            for right in range(left + 1, width):
                upper_edge = valid & (np.minimum(windows[:, left], windows[:, right]) > upper_barrier)
                lower_edge = valid & (np.minimum(-windows[:, left], -windows[:, right]) > lower_barrier)
                upper_degree[:, left] += upper_edge
                upper_degree[:, right] += upper_edge
                lower_degree[:, left] += lower_edge
                lower_degree[:, right] += lower_edge
                upper_barrier = np.maximum(upper_barrier, windows[:, right])
                lower_barrier = np.maximum(lower_barrier, -windows[:, right])
        # Recent branching hubs measure whether upward or downward shocks dominate connectivity.
        recent = slice(width // 2, width)
        upper_energy = np.square(np.maximum(upper_degree[:, recent] - 2.0, 0.0)).sum(axis=1)
        lower_energy = np.square(np.maximum(lower_degree[:, recent] - 2.0, 0.0)).sum(axis=1)
        out["upper_hub_energy"] = np.where(valid, upper_energy, np.nan)
        out["lower_hub_energy"] = np.where(valid, lower_energy, np.nan)
        out["topology"] = (out["upper_hub_energy"] - out["lower_hub_energy"]) / (
            out["upper_hub_energy"] + out["lower_hub_energy"] + 1.0
        )
        out["net_return"] = close / close.shift(width) - 1.0
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        topology = indicators["topology"].to_numpy(dtype=float)
        movement = indicators["net_return"].to_numpy(dtype=float)
        valid = (
            np.isfinite(close) & np.isfinite(ma) & np.isfinite(atr)
            & np.isfinite(topology) & np.isfinite(movement) & (atr > 0)
        )
        long_entry = valid & (close > ma) & (topology > params.topology_threshold) & (movement > 0)
        short_entry = valid & (close < ma) & (topology < -params.topology_threshold) & (movement < 0)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        pending = False
        age = 0
        entry = 0.0
        frozen_atr = 0.0
        for i in range(len(data)):
            if pending:
                # The preceding close's decision executes at this bar's open.
                entry = opening[i]
                pending = False
                if not np.isfinite(entry) or entry <= 0:
                    position = 0
                    continue
            if position:
                age += 1
                adverse_move = position * (close[i] - entry)
                stopped = np.isfinite(close[i]) and adverse_move < -params.stop_atr * frozen_atr
                if stopped or age >= params.hold_bars or not np.isfinite(close[i]):
                    position = 0
                    age = 0
                    # Preserve a flat execution bar before considering another entry.
                    continue
            elif long_entry[i] or short_entry[i]:
                position = 1 if long_entry[i] else -1
                frozen_atr = atr[i]
                pending = True
                age = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = Params()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({"atr": 2.0, "ma200": 90.0, "topology": 0.0, "net_return": 0.1}, index=index)
    ind.iloc[0, ind.columns.get_loc("topology")] = 0.8
    run = lambda d, f: GeneratedStrategy.generate_signals(None, d, f, None, params).data
    assert run(data, ind)["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    ind.loc[index[1]:, "atr"] = 100.0
    data.loc[index[2], "close"] = 96.0
    assert run(data, ind)["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    ind["ma200"] = 110.0
    ind["net_return"] = -0.1
    ind.loc[index[0], "topology"] = -0.8
    data.loc[index[2], "close"] = 104.0
    result = run(data, ind)
    assert result["signal"].tolist() == [0, -1, -1, 0, 0, 0, 0, 0]
    assert result["size"].gt(0).all()
    flat = data.assign(close=100.0)
    derived = GeneratedStrategy.indicators(None, flat, params)
    assert derived["topology"].isna().all()
    assert run(flat, derived)["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
