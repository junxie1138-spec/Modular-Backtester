from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TopologyParams:
    topology_threshold: float = 0.15
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[TopologyParams]):
    strategy_id = "gen_local-w2_1791338677"

    @classmethod
    def params_type(cls):
        return TopologyParams

    def warmup_bars(self, params: TopologyParams) -> int:
        return 15

    def indicators(self, data: pd.DataFrame, params: TopologyParams) -> pd.DataFrame:
        if not (np.isfinite(params.topology_threshold) and
                0.0 < params.topology_threshold < 1.0 and
                np.isfinite(params.trail_atr) and params.trail_atr > 0.0):
            raise ValueError("Require 0 < topology_threshold < 1 and trail_atr > 0.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0.0))
        nodes = np.column_stack([
            out["log_close"].shift(lag).to_numpy() for lag in range(12)
        ])
        valid = np.isfinite(nodes).all(axis=1)
        bowl_edges = np.zeros(len(data), dtype=float)
        dome_edges = np.zeros(len(data), dtype=float)
        # Fixed 12-node natural visibility graphs; adjacent edges cancel.
        for newer in range(10):
            for older in range(newer + 2, 12):
                below = valid.copy()
                above = valid.copy()
                for middle in range(newer + 1, older):
                    chord = nodes[:, newer] + (
                        nodes[:, older] - nodes[:, newer]
                    ) * (middle - newer) / (older - newer)
                    below &= nodes[:, middle] < chord
                    above &= nodes[:, middle] > chord
                bowl_edges += below
                dome_edges += above
        out["bowl_density"] = np.where(valid, bowl_edges / 55.0, np.nan)
        out["dome_density"] = np.where(valid, dome_edges / 55.0, np.nan)
        out["topology_balance"] = out["bowl_density"] - out["dome_density"]
        balance = out["topology_balance"]
        regime = pd.Series(np.nan, index=data.index, dtype=float)
        regime.loc[balance >= params.topology_threshold] = 1.0
        regime.loc[balance <= -params.topology_threshold] = -1.0
        out["regime"] = regime.ffill().fillna(0.0)
        previous_close = close.shift(1)
        out["return_1"] = close / previous_close.where(previous_close > 0.0) - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out.loc[previous_close.isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            balance.notna() & out["atr"].gt(0.0) &
            out["regime"].eq(1.0) & out["return_1"].gt(0.0) &
            (out["regime"].shift(1).ne(1.0) |
             out["return_1"].shift(1).le(0.0))
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: TopologyParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entry_flags = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        pending = False
        active = False
        armed = False
        entry_price = stop = peak = 0.0
        age = 0
        for i in range(len(data)):
            # A previous close's entry decision fills at today's open.
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0.0:
                    active = True
                    armed = False
                    entry_price = peak = opens[i]
                    stop = entry_price - params.trail_atr * atr[i - 1]
                    age = 0
            if active:
                age += 1
                if not (
                    np.isfinite(highs[i]) and np.isfinite(lows[i]) and
                    np.isfinite(closes[i]) and np.isfinite(atr[i]) and atr[i] > 0.0
                ):
                    active = False
                    continue
                # Test the existing stop before ratcheting at this close.
                # Stop hits request liquidation at the NEXT open, never intrabar.
                if lows[i] <= stop or age >= 4:
                    active = False
                    continue
                peak = max(peak, highs[i])
                if peak >= entry_price * 1.0075:
                    armed = True
                if armed:
                    stop = max(stop, entry_price, peak - params.trail_atr * atr[i])
                raw[i] = 1
            elif entry_flags[i] and np.isfinite(atr[i]) and atr[i] > 0.0:
                pending = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = TopologyParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=15)
    idx = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 100.2,
        "close": 100.5, "volume": 1000.0,
    }, index=idx)
    supplied = pd.DataFrame({"entry": False, "atr": 1.0}, index=idx)
    supplied.iloc[0, supplied.columns.get_loc("entry")] = True
    data.iloc[2, data.columns.get_loc("low")] = 99.9
    result = strategy.generate_signals(data, supplied, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0]
    data["low"] = 100.2
    result = strategy.generate_signals(data, supplied, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    # A volatility jump must not lower an already raised stop.
    supplied.loc[idx[2], "atr"] = 20.0
    data.loc[idx[3], "low"] = 99.9
    result = strategy.generate_signals(data, supplied, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    long_idx = pd.date_range("2020-01-01", periods=60)
    prices = 100.0 + np.sin(np.arange(60) / 3.0) + np.arange(60) * 0.05
    history = pd.DataFrame({
        "open": prices, "high": prices + 1.0, "low": prices - 1.0,
        "close": prices, "volume": 1000.0,
    }, index=long_idx)
    full = strategy.indicators(history, params)
    prefix = strategy.indicators(history.iloc[:40], params)
    pd.testing.assert_frame_equal(full.iloc[:40], prefix)
    a = strategy.generate_signals(history, full, ctx, params).data
    b = strategy.generate_signals(history.iloc[:40], prefix, ctx, params).data
    pd.testing.assert_frame_equal(a.iloc[:40], b)
    assert full["entry"].iloc[:14].eq(False).all()
    assert set(a["signal"].unique()) <= {0, 1}
    assert a["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()
