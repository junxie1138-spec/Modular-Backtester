from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    demand_window: int = 3
    capacity_window: int = 20
    capacity_quantile: float = 0.75
    overflow_multiple: float = 2.0
    atr_window: int = 14
    stop_atr: float = 1.25
    hold_days: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791319526"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        return int(max(200, params.demand_window + 2,
                       params.capacity_window + 3, params.atr_window + 1))

    def indicators(self, data, params):
        if (params.demand_window < 1 or params.capacity_window < 2
                or params.atr_window < 1 or params.hold_days not in (1, 2)
                or not 0 < params.capacity_quantile < 1
                or not np.isfinite(params.overflow_multiple)
                or params.overflow_multiple <= 0
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["returns"] = data["close"].pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan)
        out["demand"] = out["returns"].rolling(params.demand_window).sum()
        out["capacity"] = out["returns"].abs().shift(1).rolling(
            params.capacity_window).quantile(params.capacity_quantile)
        out["overflow"] = out["demand"] - params.overflow_multiple * out["capacity"]
        out["regime_ma"] = data["close"].rolling(200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        ready = np.isfinite(out[["demand", "capacity", "regime_ma", "atr"]]).all(axis=1)
        out["entry"] = (ready & (out["capacity"] > 0) & (out["atr"] > 0)
                        & (out["overflow"] > 0) & (out["overflow"].shift(1) <= 0)
                        & (out["returns"] > 0) & (data["close"] > out["regime_ma"]))
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        decision = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = pending = False
        entry_atr = stop = 0.0
        entry_bar = 0
        for i in range(len(data)):
            if pending:
                # The previous decision fills at this open; freeze its ATR.
                active = bool(np.isfinite(opens[i]) and opens[i] > 0)
                if active:
                    entry_bar = i
                    stop = opens[i] - params.stop_atr * entry_atr
                pending = False
            if active:
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or i - entry_bar + 1 >= params.hold_days):
                    active = False
                else:
                    decision[i] = 1
            elif (entries[i] and np.isfinite(closes[i]) and closes[i] > 0
                  and np.isfinite(atr[i]) and atr[i] > 0):
                decision[i] = 1
                pending = True
                entry_atr = atr[i]
        df = pd.DataFrame({"signal": decision, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 96.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = pd.DataFrame({"atr": 2.0, "entry": [True, False, False, False, False]},
                       index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "close"] = 97.0
    assert strategy.generate_signals(data, ind, ctx, params).data["signal"].tolist() == [0, 1, 0, 0, 0]
