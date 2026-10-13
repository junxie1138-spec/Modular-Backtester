from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    min_streak: int = 3
    atr_window: int = 14
    stop_atr: float = 2.0
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local_1791345774"

    @classmethod
    def params_type(cls):
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return max(params.atr_window + 1, params.min_streak + 2)

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        if params.min_streak < 1 or params.atr_window < 1 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()],
            axis=1,
        ).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["up_absorption"] = (
            (data["low"] < out["previous_close"])
            & (data["close"] > out["previous_close"])
        ).fillna(False)
        out["down_absorption"] = (
            (data["high"] > out["previous_close"])
            & (data["close"] < out["previous_close"])
        ).fillna(False)
        for direction in ("up", "down"):
            active = out[direction + "_absorption"]
            out[direction + "_streak"] = (
                active.astype(int).groupby((~active).cumsum()).cumsum()
                .clip(upper=params.min_streak)
            )
        ready = np.isfinite(out["atr"]) & (out["atr"] > 0)
        out["qualified"] = np.select(
            [ready & (out["up_streak"] >= params.min_streak),
             ready & (out["down_streak"] >= params.min_streak)],
            [1, -1], default=0,
        ).astype(int)
        out["confirmed"] = out["qualified"].where(
            out["qualified"].eq(out["qualified"].shift(1)), 0
        ).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AbsorptionParams,
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        entry_atr = np.nan
        fixed_stop = np.nan
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        for i in range(len(data)):
            if position:
                if age == 0:
                    # The preceding close requested entry; this bar supplies its open.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        position = 0
                        signals[i] = 0
                        continue
                    fixed_stop = opens[i] - position * params.stop_atr * entry_atr
                age += 1
                stopped = np.isfinite(closes[i]) and (
                    (position == 1 and closes[i] < fixed_stop)
                    or (position == -1 and closes[i] > fixed_stop)
                )
                if stopped or age >= params.hold_bars:
                    position = 0
                    age = 0
            elif confirmed[i] and np.isfinite(atr[i]) and atr[i] > 0:
                position = int(confirmed[i])
                entry_atr = atr[i]
                age = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=9)
    data = pd.DataFrame(
        {"open": 100.0, "high": 106.0, "low": 94.0,
         "close": [100., 105., 99., 97., 100., 95., 101., 103., 100.],
         "volume": 1000.0}, index=index,
    )
    indicators = pd.DataFrame(
        {"atr": 1.0, "confirmed": [1, 0, 0, 0, -1, 0, 0, 0, 0]}, index=index
    )
    result = GeneratedStrategy.generate_signals(
        None, data, indicators, None, AbsorptionParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, -1, -1, -1, 0]
    expiry = GeneratedStrategy.generate_signals(
        None, data, indicators, None, AbsorptionParams(hold_bars=2)
    ).data
    assert expiry["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
