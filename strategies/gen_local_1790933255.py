from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    graph_window: int = 18
    degree_threshold: float = 0.35
    efficiency_window: int = 8
    efficiency_threshold: float = 0.30
    atr_window: int = 14
    initial_atr: float = 2.5
    breakeven_pct: float = 0.01
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1790933255"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        return int(max(params.graph_window, params.efficiency_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        if (params.graph_window < 2 or params.efficiency_window < 1
                or params.atr_window < 1 or params.max_hold < 1
                or not 0 < params.degree_threshold <= 1
                or not 0 < params.efficiency_threshold <= 1
                or not params.initial_atr > 0 or not params.trail_atr > 0
                or not params.breakeven_pct > 0):
            raise ValueError("Invalid visibility or exit parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["log_close"] = np.log(close).replace([np.inf, -np.inf], np.nan)
        x = out["log_close"]
        ceiling = np.full(len(data), -np.inf)
        up_degree = np.zeros(len(data))
        down_degree = np.zeros(len(data))
        # Endpoint natural-visibility edges: intervening vertices must lie below the chord.
        for lag in range(1, params.graph_window):
            slope = ((x.shift(lag) - x) / lag).to_numpy()
            visible = np.isfinite(slope) & (slope > ceiling + 1e-12)
            up_degree += visible & (slope < 0)
            down_degree += visible & (slope > 0)
            ceiling = np.maximum(ceiling, slope)
        complete = x.rolling(params.graph_window).count().eq(params.graph_window)
        out["up_degree"] = pd.Series(up_degree, index=data.index).where(complete)
        out["down_degree"] = pd.Series(down_degree, index=data.index).where(complete)
        out["up_density"] = out["up_degree"] / (params.graph_window - 1)
        out["log_step"] = x.diff()
        out["net_drift"] = x.diff(params.efficiency_window)
        out["path_length"] = out["log_step"].abs().rolling(params.efficiency_window).sum()
        out["efficiency"] = out["net_drift"] / out["path_length"].replace(0, np.nan)
        prev = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"], (data["high"] - prev).abs(),
             (data["low"] - prev).abs()], axis=1
        ).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: VisibilityParams,
    ) -> SignalFrame:
        # AND: upward endpoint connectivity and independently efficient positive drift.
        eligible = (
            indicators["up_density"].ge(params.degree_threshold)
            & indicators["up_degree"].gt(indicators["down_degree"])
            & indicators["efficiency"].ge(params.efficiency_threshold)
            & indicators["atr"].gt(0)
        ).fillna(False).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        active = pending = armed = False
        entry = stop = peak = 0.0
        age = 0
        warmup = self.warmup_bars(params)
        for i in range(len(data)):
            valid = (np.isfinite([opens[i], highs[i], lows[i], closes[i], atr[i]]).all()
                     and opens[i] > 0 and closes[i] > 0 and atr[i] > 0)
            if not valid:
                active = pending = False
                continue
            if pending:
                # Observe the entry open only on its own bar, never on the decision bar.
                entry = opens[i]
                stop = max(0.0, entry - params.initial_atr * atr[i - 1])
                peak = entry
                age = 0
                active, pending, armed = True, False, False
            if active:
                age += 1
                # A touch of the previously established stop schedules next-open liquidation.
                if lows[i] <= stop:
                    active = False
                    continue
                peak = max(peak, highs[i])
                if peak >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # New ratchets use the close on this bar; no assumed high/low ordering.
                if closes[i] <= stop or age >= params.max_hold:
                    active = False
                    continue
                raw[i] = 1
            elif i >= warmup - 1 and eligible[i]:
                pending = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = VisibilityParams(graph_window=6, efficiency_window=3, atr_window=3)
    t = np.arange(40, dtype=float)
    close = 100.0 * np.exp(0.0001 * t * t)
    data = pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1,
         "close": close, "volume": 1000000.0},
        index=pd.date_range("2020-01-01", periods=len(t)),
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    full = strategy.generate_signals(data, ind, ctx, params).data
    assert ind["up_degree"].iloc[-1] == 5
    assert set(full["signal"].unique()) <= {0, 1} and full["size"].gt(0).all()
    assert full["signal"].iloc[:strategy.warmup_bars(params)].eq(0).all()
    prefix = data.iloc[:25]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(full.iloc[:25], partial)
    # Force entry, then activation and a breach of the ratcheted ATR stop.
    params = VisibilityParams(graph_window=2, efficiency_window=1, atr_window=1)
    data = data.iloc[:6].copy()
    data["open"] = 100.0
    data["close"] = [100.0, 100.0, 103.0, 102.0, 100.0, 100.0]
    data["high"] = data["close"] + 0.2
    data["low"] = data["close"] - 0.2
    ind = pd.DataFrame(
        {"up_density": [0, 1, 0, 0, 0, 0], "up_degree": 1.0,
         "down_degree": 0.0, "efficiency": 1.0, "atr": 1.0}, index=data.index,
    )
    signals = strategy.generate_signals(data, ind, ctx, params).data["signal"]
    expected = [0, 0, 1, 1, 1, 0]
    assert signals.tolist() == expected
    # Expanding ATR must never lower an already established trailing stop.
    data.loc[data.index[4:], ["close", "high", "low"]] = [100.5, 100.7, 100.3]
    ind["atr"] = [1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == expected
    # With a wide trail, the entry-price floor alone must still trigger liquidation.
    data.loc[data.index[4:], ["close", "high", "low"]] = [100.0, 100.2, 99.8]
    ind["atr"] = [1.0, 1.0, 2.0, 4.0, 4.0, 4.0]
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == expected


if __name__ == "__main__":
    _self_check()
