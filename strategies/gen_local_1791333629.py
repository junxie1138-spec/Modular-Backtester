from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    graph_window: int = 24
    far_distance: int = 5
    min_births: int = 2
    atr_period: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8
    position_min: float = 0.25
    position_max: float = 0.85

    def __post_init__(self):
        if not (2 <= self.far_distance < self.graph_window):
            raise ValueError("Require 2 <= far_distance < graph_window")
        if not (1 <= self.min_births <= self.graph_window - self.far_distance):
            raise ValueError("Invalid min_births")
        if self.atr_period < 1 or not 5 <= self.hold_bars <= 10:
            raise ValueError("Invalid ATR period or holding horizon")
        if not np.isfinite(self.stop_atr) or self.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive")
        if not 0 <= self.position_min < self.position_max <= 1:
            raise ValueError("Invalid relative-position bounds")


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791333629"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.graph_window + 1, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & (data["close"] > 0))
        valid_window = close.rolling(params.graph_window).count().eq(params.graph_window)
        blocker = pd.Series(-np.inf, index=data.index)
        far_edges = pd.Series(0.0, index=data.index)
        births = pd.Series(0.0, index=data.index)
        for distance in range(1, params.graph_window):
            past = close.shift(distance)
            # Horizontal visibility: both endpoints exceed every intervening close.
            edge = (np.minimum(close, past) > blocker) & close.notna() & past.notna()
            out[f"edge_{distance}"] = edge.astype(float)
            if distance >= params.far_distance:
                far_edges += edge.astype(float)
                # Align by historical vertex identity, not by lag number.
                previously_visible = out[f"edge_{distance - 1}"].shift(1).fillna(0).eq(1)
                births += (edge & ~previously_visible).astype(float)
            blocker = np.maximum(blocker, past)

        out["far_edges"] = far_edges.where(valid_window)
        out["edge_births"] = births.where(valid_window & valid_window.shift(1, fill_value=False))
        for distance in range(1, params.graph_window):
            out[f"edge_{distance}"] = out[f"edge_{distance}"].where(valid_window)
        out["strength"] = (
            0.5 * out["edge_births"].div(out["far_edges"].replace(0, np.nan))
            + 0.5 * (out["edge_births"] / 4.0).clip(0, 1)
        ).clip(0, 1)

        out["range_high"] = data["high"].rolling(params.graph_window).max()
        out["range_low"] = data["low"].rolling(params.graph_window).min()
        width = (out["range_high"] - out["range_low"]).replace(0, np.nan)
        out["relative_position"] = ((close - out["range_low"]) / width).clip(0, 1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - close.shift(1)).abs(),
             (data["low"] - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1).where(close.shift(1).notna() & close.notna())
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["entry"] = (
            out["edge_births"].ge(params.min_births)
            & out["relative_position"].between(params.position_min, params.position_max)
            & np.isfinite(out["atr"]) & out["atr"].gt(0)
            & np.isfinite(out["strength"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        n = len(data)
        raw_signal = np.zeros(n, dtype=int)
        raw_size = np.ones(n, dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        strength = indicators["strength"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        stop = 0.0
        age = 0
        allocation = 1.0

        for i in range(n):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    # Entry is the next-bar open; freeze the decision-bar ATR.
                    stop = opens[i] - params.stop_atr * atr[i - 1]
                    age = 0
            if active:
                age += 1
                if not np.isfinite(closes[i]) or closes[i] < stop or age >= params.hold_bars:
                    active = False
                else:
                    raw_signal[i] = 1
                    raw_size[i] = allocation
            elif entries[i] and np.isfinite(strength[i]) and np.isfinite(closes[i]) and closes[i] > 0:
                allocation = float(0.30 + 0.65 * np.clip(strength[i], 0, 1))
                raw_signal[i] = 1
                raw_size[i] = allocation
                pending = True

        df = pd.DataFrame({"signal": raw_signal, "size": raw_size}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    # Check delayed entry, fixed rather than trailing stop, and delayed exit.
    index = pd.date_range("2020-01-01", periods=14)
    close = np.full(14, 10.0)
    close[9:12] = [20.0, 9.0, 7.0]
    data = pd.DataFrame({"open": 10.0, "high": np.maximum(close, 10.0) + 1,
                         "low": np.minimum(close, 10.0) - 1,
                         "close": close, "volume": 1000.0}, index=index)
    params = GeneratedParams(graph_window=8, far_distance=3, atr_period=2)
    strategy = GeneratedStrategy()
    indicators = strategy.indicators(data, params)
    assert indicators.index.equals(data.index)
    assert not indicators["entry"].iloc[:8].any()
    controlled = pd.DataFrame({"atr": 1.0, "strength": 0.5, "entry": False}, index=index)
    controlled.loc[index[8], "entry"] = True
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    result = strategy.generate_signals(data, controlled, ctx, params).data
    assert result["signal"].iloc[:9].eq(0).all()
    assert result["signal"].iloc[9:12].eq(1).all()
    assert result["signal"].iloc[12:].eq(0).all()
    assert np.isclose(result["size"].iloc[9], 0.625)
    assert result["size"].gt(0).all()
