from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapVacuumParams:
    atr_days: int = 14
    volume_days: int = 20
    gap_atr_min: float = 0.65
    volume_ratio_max: float = 0.85
    breakeven_pct: float = 0.006
    trail_atr: float = 1.2
    initial_stop_atr: float = 2.0
    holding_days: int = 4

    def __post_init__(self):
        if self.atr_days < 1 or self.volume_days < 1:
            raise ValueError("Lookbacks must be positive.")
        if self.holding_days not in (3, 4, 5):
            raise ValueError("holding_days must be 3, 4, or 5.")
        values = (self.gap_atr_min, self.volume_ratio_max,
                  self.breakeven_pct, self.trail_atr, self.initial_stop_atr)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Thresholds and stop distances must be finite and positive.")


class GeneratedStrategy(BaseStrategy[GapVacuumParams]):
    strategy_id = "gen_local_1791243482"

    @classmethod
    def params_type(cls):
        return GapVacuumParams

    def warmup_bars(self, params):
        return int(max(200, params.atr_days + 1, params.volume_days + 1))

    def indicators(self, data, params):
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()],
            axis=1,
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_days, min_periods=params.atr_days
        ).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["previous_volume"] = data["volume"].rolling(
            params.volume_days, min_periods=params.volume_days
        ).mean().shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["gap_atr"] = out["gap"] / out["previous_atr"].replace(0, np.nan)
        out["unrepaired"] = (
            (data["close"] - out["previous_close"])
            / out["gap"].replace(0, np.nan)
        )
        out["volume_ratio"] = (
            data["volume"] / out["previous_volume"].replace(0, np.nan)
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        ready = np.isfinite(indicators).all(axis=1)
        valid_price = (
            np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
            & data[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & data["high"].ge(data[["open", "close"]].max(axis=1))
            & data["low"].le(data[["open", "close"]].min(axis=1))
        )
        eligible = (
            ready & valid_price & data["volume"].ge(0)
            & indicators["atr"].gt(0)
            & indicators["previous_atr"].gt(0)
            & indicators["volume_ratio"].le(params.volume_ratio_max)
            & indicators["unrepaired"].between(0.5, 1.0)
        )
        long_entry = (
            eligible & indicators["gap_atr"].le(-params.gap_atr_min)
            & data["close"].gt(indicators["ma200"])
        )
        short_entry = (
            eligible & indicators["gap_atr"].ge(params.gap_atr_min)
            & data["close"].lt(indicators["ma200"])
        )
        requests = np.select([long_entry, short_entry], [1, -1], default=0)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        usable = (ready & valid_price & indicators["atr"].gt(0)).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        side, age = 0, 0
        entry = None
        best, stop, armed = 0.0, 0.0, False

        for i in range(len(data)):
            if side:
                if not usable[i]:
                    side, entry = 0, None
                    continue
                if entry is None:
                    # Yesterday's decision fills at today's open.
                    entry = opens[i]
                    age, best, armed = 0, 0.0, False
                    stop = -params.initial_stop_atr * atr[i - 1]
                age += 1
                gain = side * (closes[i] - entry)
                # Stops are evaluated at closes and exit at the next open.
                if gain <= stop or age >= params.holding_days:
                    side, entry = 0, None
                    continue
                best = max(best, gain)
                armed = armed or best >= entry * params.breakeven_pct
                if armed:
                    # Favorable-distance stop only rises; a short's price stop falls.
                    stop = max(stop, 0.0, best - params.trail_atr * atr[i])
                    if gain <= stop:
                        side, entry = 0, None
                        continue
                raw[i] = side
            elif requests[i]:
                side = int(requests[i])
                entry = None
                raw[i] = side

        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), GapVacuumParams()
    index = pd.date_range("2020-01-01", periods=8)
    for side in (1, -1):
        close = np.full(8, 100.0)
        close[2:5] = 100.0 + side * np.array([1.0, 0.8, -0.1])
        data = pd.DataFrame({
            "open": 100.0, "high": 110.0, "low": 90.0,
            "close": close, "volume": 100.0,
        }, index=index)
        ind = pd.DataFrame({
            "ma200": 100.0 - side * 10.0, "atr": 1.0,
            "previous_atr": 1.0, "volume_ratio": 0.5,
            "unrepaired": 0.75, "gap_atr": 0.0,
        }, index=index)
        ind.loc[index[1], "gap_atr"] = -side
        signals = strategy.generate_signals(data, ind, None, params).data
        assert signals["signal"].tolist() == [0, 0, side, side, side, 0, 0, 0]
        assert signals["size"].gt(0).all()
        # Wider ATR cannot loosen an already-ratcheted stop.
        data.loc[index[2:5], "close"] = 100.0 + side * np.array([3.0, 2.0, 1.5])
        ind.loc[index[3], "atr"] = 10.0
        assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [
            0, 0, side, side, side, 0, 0, 0
        ]
        data["close"] = 100.0
        ind["atr"] = 1.0
        assert strategy.generate_signals(data, ind, None, params).data["signal"].tolist() == [
            0, 0, side, side, side, side, 0, 0
        ]
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()
