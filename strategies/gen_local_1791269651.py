from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoveryParams:
    ma_window: int = 20
    atr_window: int = 14
    entry_z: float = 1.75
    recovery_fraction: float = 0.30
    trailing_atr: float = 1.0
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoveryParams]):
    strategy_id = "gen_local_1791269651"

    @classmethod
    def params_type(cls) -> type[RecoveryParams]:
        return RecoveryParams

    def warmup_bars(self, params: RecoveryParams) -> int:
        return int(max(200, params.ma_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: RecoveryParams) -> pd.DataFrame:
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = close.shift(1)
        out["anchor"] = out["previous_close"].rolling(
            params.ma_window, min_periods=params.ma_window
        ).mean()
        out["sigma"] = out["previous_close"].rolling(
            params.ma_window, min_periods=params.ma_window
        ).std(ddof=0).replace(0.0, np.nan)
        out["z"] = (close - out["anchor"]) / out["sigma"]
        out["previous_z"] = (out["previous_close"] - out["anchor"]) / out["sigma"]
        out["shortfall"] = (out["anchor"] - out["previous_close"]).where(
            out["anchor"] > out["previous_close"]
        )
        out["recovery"] = (close - out["previous_close"]) / out["shortfall"]
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RecoveryParams
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = (data["close"] > indicators["ma200"]).fillna(False).to_numpy()
        valid = (
            np.isfinite(close)
            & np.isfinite(atr)
            & (atr > 0.0)
            & np.isfinite(indicators["ma200"].to_numpy(dtype=float))
        )
        entry = (
            (indicators["previous_z"] <= -params.entry_z)
            & (indicators["z"] < 0.0)
            & (indicators["recovery"] >= params.recovery_fraction)
            & (indicators["recovery"] < 1.0)
        ).fillna(False).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = np.nan
        stop = -np.inf
        for i, price in enumerate(close):
            if not valid[i]:
                active = False
                continue
            if active:
                age += 1
                # Entry fills on this bar after the final signal shift.
                # Only closes observed since that fill enter the high-water mark.
                high_water = price if age == 1 else max(high_water, price)
                stop = max(stop, high_water - params.trailing_atr * atr[i])
                if price <= stop or age >= params.hold_bars or not regime[i]:
                    active = False
                else:
                    raw[i] = 1
            elif regime[i] and entry[i]:
                active = True
                age = 0
                high_water = np.nan
                stop = -np.inf
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    prices = [100.0, 100.0, 105.0, 102.0, 101.0]
    data = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices,
         "close": prices, "volume": 1000.0}, index=index
    )
    indicators = pd.DataFrame(
        {"ma200": 90.0, "atr": [1.0, 1.0, 1.0, 5.0, 1.0],
         "previous_z": [-2.0, 0.0, 0.0, 0.0, 0.0],
         "z": [-1.0, 0.0, 0.0, 0.0, 0.0],
         "recovery": [0.5, 0.0, 0.0, 0.0, 0.0]}, index=index
    )
    strategy = GeneratedStrategy()
    for hold, expected in [(2, [0, 1, 1, 0, 0]), (10, [0, 1, 1, 1, 0])]:
        result = strategy.generate_signals(
            data, indicators, None, RecoveryParams(hold_bars=hold)
        ).data
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0.0).all()
