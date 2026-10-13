from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    pressure_bars: int = 5
    rank_bars: int = 63
    entry_percentile: float = 0.8
    atr_bars: int = 14
    breakeven_pct: float = 0.012
    trail_atr: float = 2.0
    initial_stop_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791216745"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(200, params.pressure_bars + params.rank_bars - 1,
                       params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        span = (data["high"] - data["low"]).replace(0.0, np.nan)
        out["selling_fraction"] = ((data["high"] - close) / span).clip(0, 1).fillna(0.5)
        out["selling_volume"] = out["selling_fraction"] * data["volume"]
        out["pressure"] = (
            out["selling_volume"].rolling(params.pressure_bars).sum()
            / data["volume"].rolling(params.pressure_bars).sum().replace(0.0, np.nan)
        )
        out["pressure_rank"] = out["pressure"].rolling(params.rank_bars).rank(pct=True)
        out["ma200"] = close.rolling(200).mean()
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        day = data.index.day
        weekday = data.index.dayofweek
        out["expiry_unwind"] = (
            data.index.month.isin([3, 6, 9, 12])
            & (((weekday == 0) & (day >= 18) & (day <= 24))
               | ((weekday == 1) & (day >= 19) & (day <= 25)))
        )
        out["entry"] = (
            out["expiry_unwind"]
            & (out["pressure_rank"] >= params.entry_percentile)
            & (close > out["ma200"])
            & (close > out["previous_close"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = stop = peak = entry_atr = 0.0
        age = 0
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # Entry is the next bar's open; exits decide at the close.
                    entry = peak = opens[i]
                    stop = entry - params.initial_stop_atr * entry_atr
                    active = True
                    armed = False
                    age = 0
            if active:
                age += 1
                if not (np.isfinite(close[i]) and np.isfinite(atr[i])
                        and atr[i] > 0 and np.isfinite(regime[i])):
                    active = False
                    continue
                peak = max(peak, close[i])
                if peak >= entry * (1.0 + params.breakeven_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                if close[i] <= stop or close[i] <= regime[i] or age >= params.hold_bars:
                    active = False
                else:
                    raw[i] = 1
            elif entries[i] and np.isfinite(atr[i]) and atr[i] > 0:
                entry_atr = atr[i]
                pending = True
                raw[i] = 1
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-01", periods=260)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    data.iloc[202, data.columns.get_loc("close")] = 102.0
    data.iloc[203, data.columns.get_loc("close")] = 104.0
    data.iloc[204, data.columns.get_loc("close")] = 101.0
    ind = pd.DataFrame({"atr": 1.0, "ma200": 90.0, "entry": False}, index=index)
    ind.iloc[200, ind.columns.get_loc("entry")] = True
    strategy = GeneratedStrategy()
    params = Params()
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[200] == 0
    assert result["signal"].iloc[201:205].eq(1).all()
    assert result["signal"].iloc[205] == 0
    assert result["signal"].isin([0, 1]).all()
    assert result["size"].gt(0).all()
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert not actual["entry"].iloc[:199].any()
    assert strategy.warmup_bars(Params(rank_bars=250)) == 254
    prefix = strategy.generate_signals(data.iloc[:204], ind.iloc[:204], None, params).data
    assert result.iloc[:204].equals(prefix)


if __name__ == "__main__":
    _self_check()
