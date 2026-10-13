from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 18
    max_complexity: float = 0.30
    motif_edge: float = 0.20
    movement_window: int = 8
    movement_edge: float = 0.10
    atr_window: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.0
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790919970"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.pattern_window + 3, params.movement_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.pattern_window, params.movement_window, params.atr_window, params.max_hold) < 1:
            raise ValueError("Windows and holding period must be positive.")
        if not (0 <= params.max_complexity <= 1 and 0 < params.motif_edge <= 1 and 0 < params.movement_edge <= 1):
            raise ValueError("Complexity and directional thresholds are out of range.")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError("Breakeven threshold and ATR multiplier must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["close_lag1"] = close.shift(1)
        out["close_lag2"] = close.shift(2)
        a, b, c = out["close_lag2"], out["close_lag1"], close
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        out["motif"] = (4 * a.gt(b).astype(int) + 2 * a.gt(c).astype(int) + b.gt(c).astype(int)).where(valid)
        out["motif_lag1"] = out["motif"].shift(1)
        out["transition"] = 8 * out["motif_lag1"] + out["motif"]
        out["valid_transition_count"] = out["transition"].notna().astype(float).rolling(params.pattern_window).sum()
        codes = (0, 1, 3, 4, 6, 7)
        repertoire = pd.Series(0.0, index=data.index)
        for left in codes:
            for right in codes:
                code = 8 * left + right
                name = f"transition_count_{code}"
                out[name] = out["transition"].eq(code).astype(float).rolling(params.pattern_window).sum()
                repertoire += out[name].gt(0).astype(float)
        ready = out["valid_transition_count"].eq(params.pattern_window)
        out["complexity"] = (repertoire / 36.0).where(ready)
        out["motif_balance"] = ((out["transition_count_0"] - out["transition_count_63"]) / params.pattern_window).where(ready)
        out["high_change"] = data["high"].diff()
        out["low_change"] = -data["low"].diff()
        up, down = out["high_change"], out["low_change"]
        out["positive_dm"] = up.where(up.gt(down) & up.gt(0), 0.0).where(up.notna() & down.notna())
        out["negative_dm"] = down.where(down.gt(up) & down.gt(0), 0.0).where(up.notna() & down.notna())
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["close_lag1"]).abs()
        out["low_gap"] = (data["low"] - out["close_lag1"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["movement_range"] = out["true_range"].rolling(params.movement_window).sum()
        out["positive_dm_sum"] = out["positive_dm"].rolling(params.movement_window).sum()
        out["negative_dm_sum"] = out["negative_dm"].rolling(params.movement_window).sum()
        out["movement_balance"] = (out["positive_dm_sum"] - out["negative_dm_sum"]) / out["movement_range"].replace(0, np.nan)
        simple = out["complexity"].le(params.max_complexity)
        out["entry_direction"] = np.select(
            [simple & out["motif_balance"].ge(params.motif_edge) & out["movement_balance"].ge(params.movement_edge),
             simple & out["motif_balance"].le(-params.motif_edge) & out["movement_balance"].le(-params.movement_edge)],
            [1, -1], default=0,
        ).astype(int)
        out.loc[~out["atr"].gt(0), "entry_direction"] = 0
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        target = np.zeros(n, dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        side = 0
        entry = np.nan
        stop = np.nan
        peak = np.nan
        armed = False
        age = 0
        for i in range(n):
            if side:
                # The previous close's instruction fills at this bar's open.
                if not np.isfinite(entry):
                    if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0 or not np.isfinite(atr[i - 1]) or atr[i - 1] <= 0:
                        side = 0
                        target[i] = 0
                        continue
                    entry = opens[i]
                    peak = side * entry
                    stop = peak - params.trail_atr * atr[i - 1]
                    armed = False
                    age = 0
                age += 1
                price = side * closes[i]
                # Favorable-price coordinates make the stop increase for both sides.
                # Stops are observed at close; exit instructions fill next open.
                if not np.isfinite(price) or price <= stop or age >= params.max_hold:
                    side = 0
                    entry = np.nan
                else:
                    peak = max(peak, price)
                    armed = armed or peak >= side * entry + entry * params.breakeven_pct
                    if armed:
                        stop = max(stop, side * entry)
                        if np.isfinite(atr[i]) and atr[i] > 0:
                            stop = max(stop, peak - params.trail_atr * atr[i])
                        if price <= stop:
                            side = 0
                            entry = np.nan
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                side = int(entries[i])
                entry = np.nan
            target[i] = side
        df = pd.DataFrame({"signal": target, "size": np.full(n, 0.95, dtype=float)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = Params()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    for direction in (1, -1):
        close = 100.0 + direction * np.array([0.0, 2.0, 4.0, 1.0, 0.0])
        data = pd.DataFrame({"open": np.full(5, 100.0), "high": close + 1.0, "low": close - 1.0, "close": close, "volume": np.full(5, 1000.0)}, index=pd.date_range("2020-01-01", periods=5))
        ind = pd.DataFrame({"atr": np.ones(5), "entry_direction": [direction, 0, 0, 0, 0]}, index=data.index)
        result = strategy.generate_signals(data, ind, ctx, params).data
        assert result["signal"].tolist() == [0, direction, direction, direction, 0]
        assert result["size"].gt(0).all()
    close = 100.0 + np.arange(80, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": np.full(80, 1000.0)}, index=pd.date_range("2020-01-01", periods=80))
    ind = strategy.indicators(data, params)
    assert ind["entry_direction"].iloc[30] == 1
    signals = strategy.generate_signals(data, ind, ctx, params).data
    prefix = data.iloc[:50]
    shorter = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(signals.iloc[:50], shorter)
    assert signals["signal"].isin([-1, 0, 1]).all()
    assert signals["signal"].iloc[0] == 0
    assert ind["entry_direction"].iloc[:params.pattern_window + 2].eq(0).all()
    empty = data.iloc[:0]
    assert strategy.generate_signals(empty, strategy.indicators(empty, params), ctx, params).data.empty


if __name__ == "__main__":
    _self_check()
