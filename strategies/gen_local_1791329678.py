from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    slope_window: int = 6
    innovation_window: int = 20
    atr_window: int = 14
    entry_z: float = 0.9
    stop_atr: float = 1.25
    hold_days: int = 2
    size_floor: float = 0.25
    entry_slippage_bps: float = 5.0

    def __post_init__(self):
        if min(self.slope_window, self.innovation_window) < 2 or self.atr_window < 1:
            raise ValueError("Invalid lookback")
        if self.hold_days not in (1, 2):
            raise ValueError("hold_days must be 1 or 2")
        if not all(np.isfinite(x) for x in (self.entry_z, self.stop_atr, self.size_floor, self.entry_slippage_bps)):
            raise ValueError("Parameters must be finite")
        if self.entry_z <= 0 or self.stop_atr <= 0 or not 0 < self.size_floor <= 1:
            raise ValueError("Invalid threshold, stop, or size")
        if not 0 <= self.entry_slippage_bps < 10000:
            raise ValueError("Invalid entry slippage")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791329678"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.atr_window + 1, params.slope_window + params.innovation_window + 3))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        # Finite-window latent slope; innovation measures its new acceleration.
        out["slope"] = data["close"].diff().rolling(params.slope_window).mean()
        out["innovation"] = out["slope"].diff()
        out["innovation_scale"] = out["innovation"].rolling(params.innovation_window).std(ddof=0).shift(1)
        out["z"] = out["innovation"] / out["innovation_scale"].where(out["innovation_scale"] > 0)
        out["gap"] = data["open"] - previous_close
        out["body"] = data["close"] - data["open"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        z = indicators["z"]
        direction = np.sign(z.fillna(0))
        ready = indicators[["atr", "slope", "z", "gap", "body"]].notna().all(axis=1)
        setup = (ready & (indicators["atr"] > 0) & (z.abs() >= params.entry_z)
                 & (direction * indicators["slope"] > 0)
                 & (direction * indicators["body"] > 0)
                 & (direction * indicators["gap"] < 0))
        candidates = direction.where(setup, 0).astype(int).to_numpy()
        conviction = ((z.abs() / params.entry_z - 1) / 2).clip(0, 1).fillna(0)
        entry_sizes = (params.size_floor + (1 - params.size_floor) * conviction).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.size_floor, dtype=float)
        position, age = 0, 0
        entry_price, stop_distance, held_size = np.nan, np.nan, params.size_floor
        for i in range(len(data)):
            if position:
                # The previous close's decision fills at THIS bar's open.
                if age == 0:
                    entry_price = opens[i] * (1 + position * params.entry_slippage_bps / 10000)
                age += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                stopped = valid and position * (closes[i] - entry_price) < -stop_distance
                if not valid or stopped or age >= params.hold_days:
                    position = 0
                # Do not re-enter on the close that orders an exit.
            elif candidates[i] and np.isfinite(closes[i]) and closes[i] > 0:
                position = int(candidates[i])
                age = 0
                stop_distance = params.stop_atr * atr[i]
                held_size = float(entry_sizes[i])
            signals[i] = position
            sizes[i] = held_size if position else params.size_floor
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.size_floor).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params(stop_atr=1.0)
    index = pd.date_range("2020-01-01", periods=4)
    data = pd.DataFrame({"open": 10.0, "high": 13.0, "low": 7.0,
                         "close": [10.0, 12.0, 10.5, 10.0], "volume": 100.0}, index=index)
    ind = pd.DataFrame({"atr": 1.0, "slope": 1.0, "z": [2.0, 0.0, 0.0, 0.0],
                        "gap": -0.5, "body": 1.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    run = lambda d, x: strategy.generate_signals(d, x, ctx, params).data
    fixed = run(data, ind)
    assert fixed["signal"].tolist() == [0, 1, 1, 0]
    shocked = data.copy()
    shocked.loc[index[1], "close"] = 8.0
    assert run(shocked, ind)["signal"].tolist() == [0, 1, 0, 0]
    short_ind = ind.copy()
    short_ind[["slope", "z", "gap", "body"]] *= -1
    shocked.loc[index[1], "close"] = 12.0
    assert run(shocked, short_ind)["signal"].tolist() == [0, -1, 0, 0]
    weak = ind.copy()
    weak.loc[index[0], "z"] = params.entry_z
    assert run(data, weak)["size"].iloc[1] < fixed["size"].iloc[1]
    assert run(data.iloc[:2], ind.iloc[:2]).equals(fixed.iloc[:2])
    flat = data.assign(close=10.0, open=10.0, high=10.0, low=10.0)
    result = run(flat, strategy.indicators(flat, params))
    assert result["signal"].eq(0).all() and result["size"].gt(0).all()
    assert run(flat.iloc[:0], strategy.indicators(flat.iloc[:0], params)).empty


if __name__ == "__main__":
    _self_check()
