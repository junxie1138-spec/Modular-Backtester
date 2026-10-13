from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_bars: int = 30
    cohort_bars: int = 5
    entry_z: float = 1.5
    atr_bars: int = 14
    stop_atr: float = 2.0
    min_hold: int = 5
    max_hold: int = 10


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791221891"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.ma_bars + 2 * params.cohort_bars + 1, params.atr_bars + 1))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if not (params.ma_bars >= 2 and params.cohort_bars >= 1 and params.atr_bars >= 1
                and params.entry_z > 0 and params.stop_atr > 0
                and 1 <= params.min_hold <= params.max_hold):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        out["ma"] = close.rolling(params.ma_bars).mean()
        out["std"] = close.rolling(params.ma_bars).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0.0, np.nan)
        out["previous_close"] = close.shift(1)
        out["tr"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["tr"].rolling(params.atr_bars).mean()
        out["previous_z"] = out["z"].shift(1)
        for name, direction in (("long", -1), ("short", 1)):
            stressed = direction * out["z"] >= params.entry_z
            out[name + "_stressed"] = stressed
            out[name + "_arrival"] = stressed & ~stressed.shift(1, fill_value=False)
            out[name + "_fraction"] = stressed.astype(float).rolling(params.cohort_bars).mean()
            out[name + "_arrivals"] = out[name + "_arrival"].astype(float).rolling(params.cohort_bars).sum()
            out[name + "_prior_arrivals"] = out[name + "_arrivals"].shift(params.cohort_bars)
            out[name + "_exhaustion"] = (
                stressed
                & (out[name + "_fraction"] >= 0.6)
                & (out[name + "_arrivals"] < out[name + "_prior_arrivals"])
                & (direction * out["z"] < direction * out["previous_z"])
            ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        n = len(data)
        signals = np.zeros(n, dtype=int)
        sizes = np.ones(n, dtype=float)
        close = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        z = indicators["z"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        position = 0
        entry_bar = -1
        stop = np.nan
        entry_atr = 0.0
        exposure = 1.0
        ready_from = self.warmup_bars(params) - 1
        for i in range(n):
            valid = np.isfinite(close[i]) and close[i] > 0
            if position:
                # The prior close's decision fills at this bar's open.
                if i == entry_bar + 1:
                    if np.isfinite(opens[i]) and opens[i] > 0:
                        stop = opens[i] - position * params.stop_atr * entry_atr
                    else:
                        position = 0
                age = i - entry_bar
                stopped = valid and np.isfinite(stop) and position * (close[i] - stop) <= 0
                reverted = np.isfinite(z[i]) and position * z[i] >= 0
                if (not valid or stopped or age >= params.max_hold
                        or (age >= params.min_hold and reverted)):
                    position = 0
            elif i >= ready_from and valid and np.isfinite(z[i]) and np.isfinite(atr[i]) and atr[i] > 0:
                name = "long" if z[i] < 0 else "short"
                exhausted = indicators[name + "_exhaustion"].iat[i]
                fraction = indicators[name + "_fraction"].iat[i]
                if pd.notna(exhausted) and bool(exhausted) and np.isfinite(fraction):
                    position = 1 if z[i] < 0 else -1
                    entry_bar = i
                    entry_atr = atr[i]
                    stop = np.nan
                    exposure = float(np.clip(abs(z[i]) / (2 * params.entry_z) * fraction, 0.25, 1.0))
            signals[i] = position
            sizes[i] = exposure if position else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params(ma_bars=2, cohort_bars=1, atr_bars=1, min_hold=1, max_hold=3)
    index = pd.date_range("2020-01-01", periods=10)
    for direction in (1, -1):
        close = np.full(10, 100.0)
        close[6] = 100.0 - 3.0 * direction
        data = pd.DataFrame({"open": 100.0, "high": 104.0, "low": 96.0,
                             "close": close, "volume": 1000.0}, index=index)
        ind = strategy.indicators(data, params)
        ind["z"] = -2.0 * direction
        ind["atr"] = 1.0
        ind["long_exhaustion"] = False
        ind["short_exhaustion"] = False
        name = "long" if direction == 1 else "short"
        ind.loc[index[4], name + "_exhaustion"] = True
        ind[name + "_fraction"] = 0.75
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].tolist() == [0, 0, 0, 0, 0, direction, direction, 0, 0, 0]
        assert np.isclose(result["size"].iloc[5], 0.5)
        assert (result["size"] > 0).all()
        data["close"] = 100.0
        result = strategy.generate_signals(data, ind, None, params).data
        assert result["signal"].iloc[7] == direction and result["signal"].iloc[8] == 0
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 0.0}, index=index)
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), None, params).data
    assert (result["signal"] == 0).all()


if __name__ == "__main__":
    _self_check()
