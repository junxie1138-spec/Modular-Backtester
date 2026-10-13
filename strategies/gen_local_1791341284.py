from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pattern_window: int = 30
    recurrence_min: float = 0.45
    diversity_min: float = 0.65
    momentum_bars: int = 8
    efficiency_min: float = 0.30
    atr_bars: int = 14
    breakeven_pct: float = 0.015
    trail_atr: float = 2.5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791341284"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.pattern_window + 6,
                       params.momentum_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.pattern_window, params.momentum_bars, params.atr_bars, params.max_hold) < 1:
            raise ValueError("Lookbacks and holding limit must be positive")
        if not (0 <= params.recurrence_min <= 1 and 0 <= params.diversity_min <= 1
                and 0 <= params.efficiency_min <= 1):
            raise ValueError("Probability and efficiency thresholds must be in [0, 1]")
        if params.breakeven_pct <= 0 or params.trail_atr <= 0:
            raise ValueError("Stop parameters must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["return"] = close.pct_change(fill_method=None)
        triples = np.column_stack([out["return"].shift(2), out["return"].shift(1), out["return"]])
        valid = np.isfinite(triples).all(axis=1)
        order = np.argsort(triples, axis=1, kind="stable")
        # Each ordering maps to a distinct code; stable sorting resolves ties chronologically.
        out["pattern"] = pd.Series(order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2],
                                   index=data.index).where(valid)
        out["previous_pattern"] = out["pattern"].shift(3)
        out["transition"] = (out["previous_pattern"] * 27 + out["pattern"]).where(
            out["previous_pattern"].notna() & out["pattern"].notna())
        w = params.pattern_window
        out["valid_count"] = out["transition"].notna().astype(float).rolling(w).sum()
        out["recurrence"] = 0.0
        out["diversity"] = 1.0
        codes = (5, 7, 11, 15, 19, 21)
        for a in codes:
            marginal = ((out["previous_pattern"] == a) & out["transition"].notna()).astype(float).rolling(w).mean()
            out[f"p_{a}"] = marginal
            out["diversity"] -= marginal.pow(2)
            for b in codes:
                joint = (out["transition"] == a * 27 + b).astype(float).rolling(w).mean()
                out[f"p_{a}_{b}"] = joint
                out["recurrence"] += joint.pow(2).div(marginal.replace(0, np.nan)).fillna(0)
        out.loc[out["valid_count"] != w, ["recurrence", "diversity"]] = np.nan
        out["ma200"] = close.rolling(200).mean()
        out["change"] = close.diff()
        out["displacement"] = close.diff(params.momentum_bars)
        out["path"] = out["change"].abs().rolling(params.momentum_bars).sum()
        out["efficiency"] = out["displacement"].abs().div(out["path"].replace(0, np.nan))
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs()], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        ready = (out["recurrence"].ge(params.recurrence_min)
                 & out["diversity"].ge(params.diversity_min)
                 & out["efficiency"].ge(params.efficiency_min)
                 & out["atr"].gt(0))
        out["candidate"] = np.select([
            ready & close.gt(out["ma200"]) & out["displacement"].gt(0),
            ready & close.lt(out["ma200"]) & out["displacement"].lt(0)], [1, -1], default=0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame({"signal": np.zeros(len(data), dtype=int), "size": 0.95}, index=data.index)
        side, age = 0, 0
        entry, stop, entry_atr = np.nan, np.nan, np.nan
        armed = False
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            atr = float(indicators["atr"].iloc[i])
            ma = float(indicators["ma200"].iloc[i])
            if side:
                # The previous close's target is filled at this bar's open.
                if age == 0:
                    entry = float(data["open"].iloc[i])
                    stop = side * entry - params.trail_atr * entry_atr
                age += 1
                if not (np.isfinite(entry) and entry > 0 and np.isfinite(close)
                        and np.isfinite(atr) and atr > 0 and np.isfinite(ma)):
                    side = 0
                elif (side * close <= stop or age >= params.max_hold
                      or side * (close - ma) <= 0):
                    side = 0
                else:
                    if side * (close / entry - 1.0) >= params.breakeven_pct:
                        armed = True
                    if armed:
                        # Profit-coordinate stop only increases, for longs AND shorts.
                        stop = max(stop, side * entry, side * close - params.trail_atr * atr)
                # A stopped position cannot reopen on the same decision bar.
            elif int(indicators["candidate"].iloc[i]) != 0:
                side = int(indicators["candidate"].iloc[i])
                age, armed = 0, False
                entry_atr = atr
            df.iloc[i, df.columns.get_loc("signal")] = side
        # ponytail: close-confirmed stops; intraday execution requires an execution-engine stop order.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    p = Params()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame({"open": [100, 100, 102, 99, 100, 100],
                         "high": [101, 103, 103, 101, 101, 101],
                         "low": [99, 99, 98, 98, 99, 99],
                         "close": [100, 102, 99, 100, 100, 100],
                         "volume": 1000}, index=index)
    ind = pd.DataFrame({"candidate": [1, 0, 0, 0, 0, 0], "atr": 1.0, "ma200": 90.0}, index=index)
    signals = strategy.generate_signals(data, ind, None, p).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0, 0]
    mirror = data.copy()
    for column in ("open", "high", "low", "close"):
        mirror[column] = 200 - data[column]
    mirror["high"], mirror["low"] = mirror["low"].copy(), mirror["high"].copy()
    ind["candidate"], ind["ma200"] = [-1, 0, 0, 0, 0, 0], 110.0
    assert strategy.generate_signals(mirror, ind, None, p).data["signal"].tolist() == [0, -1, -1, 0, 0, 0]
    derived = strategy.indicators(data, p)
    assert derived["candidate"].eq(0).all()
    assert strategy.warmup_bars(p) >= 200


if __name__ == "__main__":
    _self_check()
