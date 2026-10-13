from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 32
    atr_window: int = 14
    topology_threshold: float = 0.30
    edge_fraction: float = 0.35
    breakeven_pct: float = 0.02
    trail_atr: float = 2.5
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791325950"

    @classmethod
    def params_type(cls) -> type[VisibilityParams]:
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        if params.graph_window < 5 or params.atr_window < 1 or params.max_hold < 1:
            raise ValueError("Lookbacks and holding period must be positive; graph_window >= 5.")
        if not (0 < params.topology_threshold <= 1 and 0 < params.edge_fraction < 0.5):
            raise ValueError("Invalid topology or relative-position threshold.")
        if not (np.isfinite(params.breakeven_pct) and params.breakeven_pct > 0
                and np.isfinite(params.trail_atr) and params.trail_atr > 0):
            raise ValueError("Exit parameters must be finite and positive.")
        return int(max(200, params.graph_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        out["log_close"] = np.log(close.where(close > 0))
        out["previous_close"] = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat(
            [(data["high"] - data["low"]),
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["range_low"] = data["low"].rolling(
            params.graph_window, min_periods=params.graph_window
        ).min()
        out["range_high"] = data["high"].rolling(
            params.graph_window, min_periods=params.graph_window
        ).max()
        span = (out["range_high"] - out["range_low"]).replace(0.0, np.nan)
        out["relative_position"] = (close - out["range_low"]) / span

        values = out["log_close"].to_numpy()
        count = len(data)
        upper = np.zeros(count)
        lower = np.zeros(count)
        greatest = np.full(count, -np.inf)
        least = np.full(count, np.inf)
        # Natural visibility: an endpoint edge exists iff its backward slope
        # exceeds every intervening slope. Reflection reverses that inequality.
        # Exclude lags 1-3 so the score measures remote graph connections.
        for lag in range(1, params.graph_window):
            slopes = np.full(count, np.nan)
            slopes[lag:] = (values[:-lag] - values[lag:]) / lag
            valid = np.isfinite(slopes)
            if lag >= 4:
                upper += valid & (slopes > greatest)
                lower += valid & (slopes < least)
            greatest = np.where(valid, np.maximum(greatest, slopes), greatest)
            least = np.where(valid, np.minimum(least, slopes), least)
        complete = out["log_close"].rolling(
            params.graph_window, min_periods=params.graph_window
        ).count().eq(params.graph_window)
        out["upper_degree"] = pd.Series(upper, index=data.index).where(complete)
        out["lower_degree"] = pd.Series(lower, index=data.index).where(complete)
        out["topology_balance"] = (
            out["lower_degree"] - out["upper_degree"]
        ) / (params.graph_window - 4)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilityParams
    ) -> SignalFrame:
        self.warmup_bars(params)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        topology = indicators["topology_balance"].to_numpy(dtype=float)
        location = indicators["relative_position"].to_numpy(dtype=float)
        ready = (
            np.isfinite(close) & (close > 0) & np.isfinite(atr) & (atr > 0)
            & np.isfinite(ma) & np.isfinite(topology) & np.isfinite(location)
        )
        candidates = np.zeros(len(data), dtype=int)
        candidates[ready & (close > ma) & (location <= params.edge_fraction)
                   & (topology >= params.topology_threshold)] = 1
        candidates[ready & (close < ma) & (location >= 1 - params.edge_fraction)
                   & (topology <= -params.topology_threshold)] = -1

        decisions = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        entry = 0.0
        stop = -np.inf
        armed = False
        entry_atr = 0.0
        for i in range(len(data)):
            if position:
                # A decision made at the preceding close enters at this open.
                if held == 0:
                    entry = opens[i]
                    if not np.isfinite(entry) or entry <= 0:
                        position = 0
                        continue
                    stop = position * entry - params.trail_atr * entry_atr
                held += 1
                signed_close = position * close[i]
                # Stops are checked at the close; exit orders fill next bar.
                # In signed-price coordinates the stop only ever moves up,
                # corresponding to a rising long stop or falling short stop.
                if (not ready[i] or signed_close <= stop or held >= params.max_hold):
                    position = 0
                    held = 0
                    armed = False
                    continue
                if position * (close[i] / entry - 1.0) >= params.breakeven_pct:
                    armed = True
                if armed:
                    stop = max(stop, position * entry,
                               signed_close - params.trail_atr * atr[i])
                decisions[i] = position
            elif candidates[i]:
                position = int(candidates[i])
                entry_atr = atr[i]
                held = 0
                armed = False
                decisions[i] = position

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams()
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    close = np.full(240, 100.0)
    close[201] = 103.0
    close[211] = 97.0
    data = pd.DataFrame(
        {"open": 100.0, "high": np.maximum(close, 100.0) + 1.0,
         "low": np.minimum(close, 100.0) - 1.0, "close": close,
         "volume": 1000000.0}, index=index
    )
    derived = strategy.indicators(data, params)
    pd.testing.assert_frame_equal(
        derived.iloc[:220], strategy.indicators(data.iloc[:220], params)
    )
    curve = data.iloc[:40].copy()
    curve["close"] = np.exp(0.001 * np.arange(40) ** 2)
    convex = strategy.indicators(curve, params)
    assert convex["upper_degree"].iloc[-1] == params.graph_window - 4
    assert convex["lower_degree"].iloc[-1] == 0
    curve["close"] = 1.0 / curve["close"]
    concave = strategy.indicators(curve, params)
    assert concave["lower_degree"].iloc[-1] == params.graph_window - 4
    assert concave["upper_degree"].iloc[-1] == 0
    assert strategy.warmup_bars(params) >= 200
    assert derived["ma200"].iloc[:199].isna().all()
    fixture = pd.DataFrame(
        {"atr": 1.0, "ma200": 90.0, "topology_balance": 0.0,
         "relative_position": 0.5}, index=index
    )
    fixture.iloc[200, fixture.columns.get_loc("topology_balance")] = 0.8
    fixture.iloc[200, fixture.columns.get_loc("relative_position")] = 0.1
    fixture.loc[index[210]:, "ma200"] = 110.0
    fixture.iloc[210, fixture.columns.get_loc("topology_balance")] = -0.8
    fixture.iloc[210, fixture.columns.get_loc("relative_position")] = 0.9
    result = strategy.generate_signals(data, fixture, None, params).data
    assert result["signal"].iloc[[200, 201, 202, 203]].tolist() == [0, 1, 1, 0]
    assert result["signal"].iloc[[210, 211, 212, 213]].tolist() == [0, -1, -1, 0]
    pd.testing.assert_frame_equal(
        result.iloc[:212],
        strategy.generate_signals(data.iloc[:212], fixture.iloc[:212], None, params).data
    )
    assert result["signal"].isin([-1, 0, 1]).all()
    assert (result["size"] > 0).all()
    quiet = data.copy()
    quiet["close"] = 100.0
    fixture.loc[index[210]:, "topology_balance"] = 0.0
    timed = strategy.generate_signals(quiet, fixture, None, params).data["signal"]
    assert timed.iloc[201:219].eq(1).all() and timed.iloc[219] == 0


if __name__ == "__main__":
    _self_check()
