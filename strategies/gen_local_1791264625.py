from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    peak_window: int = 40
    atr_window: int = 14
    residence_window: int = 12
    deep_atr: float = 2.5
    repair_floor_atr: float = 0.8
    regime_on: float = 0.25
    regime_off: float = 0.60
    trail_atr: float = 1.2
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791264625"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        if min(params.peak_window, params.atr_window, params.residence_window) < 2:
            raise ValueError("Lookbacks must be at least two.")
        if not (0 < params.repair_floor_atr < params.deep_atr and
                0 <= params.regime_on < params.regime_off <= 1 and
                np.isfinite(params.deep_atr) and
                np.isfinite(params.trail_atr) and params.trail_atr > 0 and
                params.hold_bars in (1, 2)):
            raise ValueError("Invalid thresholds or holding horizon.")
        return int(max(params.peak_window, params.atr_window + 1) +
                   params.residence_window + 2)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().where(lambda x: x > 0)
        out["peak"] = data["close"].rolling(
            params.peak_window, min_periods=params.peak_window
        ).max()
        out["depth"] = (out["peak"] - data["close"]) / out["atr"]
        out["deep"] = out["depth"].ge(params.deep_atr).astype(float).where(
            out["depth"].notna()
        )
        out["deep_share"] = out["deep"].rolling(
            params.residence_window, min_periods=params.residence_window
        ).mean()
        out["repair"] = (
            data["close"].gt(out["previous_close"]) &
            out["depth"].lt(out["depth"].shift(1)) &
            out["depth"].ge(params.repair_floor_atr)
        ).fillna(False)
        out["confirmed"] = (
            out["repair"] &
            out["repair"].shift(1, fill_value=False) &
            out["depth"].shift(2).ge(params.deep_atr)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params,
    ) -> SignalFrame:
        self.warmup_bars(params)
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        share = indicators["deep_share"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        valid = np.isfinite(np.column_stack([
            close, atr, share, indicators["depth"].to_numpy(dtype=float)
        ])).all(axis=1) & (close > 0) & (atr > 0)
        signals = np.zeros(len(data), dtype=int)
        favorable = False
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            if not valid[i]:
                favorable = active = False
                age = 0
                high_water = stop = -np.inf
                continue
            if share[i] <= params.regime_on:
                favorable = True
            elif share[i] >= params.regime_off:
                favorable = False
            if active:
                # The preceding decision fills today; start at today's close.
                age += 1
                high_water = max(high_water, close[i])
                stop = max(stop, high_water - params.trail_atr * atr[i])
                if close[i] <= stop or age >= params.hold_bars:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
                else:
                    signals[i] = 1
                continue
            if favorable and confirmed[i]:
                active = True
                age = 0
                high_water = stop = -np.inf
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    prices = [100.0] * 60 + [95.0, 96.0, 97.0, 98.0, 97.0] + [100.0] * 15
    close = pd.Series(prices, index=pd.date_range("2020-01-01", periods=len(prices)))
    data = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1000.0,
    })
    strategy = GeneratedStrategy()
    params = Params()
    derived = strategy.indicators(data, params)
    result = strategy.generate_signals(data, derived, None, params).data
    assert derived["confirmed"].iloc[62]
    assert result["signal"].iloc[62:66].tolist() == [0, 1, 1, 0]
    assert set(result["signal"].unique()) <= {0, 1}
    assert result["size"].gt(0).all()
    prefix = data.iloc[:64]
    earlier = strategy.generate_signals(
        prefix, strategy.indicators(prefix, params), None, params
    ).data
    pd.testing.assert_frame_equal(result.iloc[:64], earlier)
