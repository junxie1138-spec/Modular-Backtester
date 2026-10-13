from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeHandoffParams:
    window: int = 6
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[RangeHandoffParams]):
    strategy_id = "gen_local_1790906357"

    @classmethod
    def params_type(cls) -> type[RangeHandoffParams]:
        return RangeHandoffParams

    @staticmethod
    def warmup_bars(params: RangeHandoffParams) -> int:
        return int(max(15, params.window + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeHandoffParams) -> pd.DataFrame:
        if params.window < 2 or not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("window must be >= 2 and trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["change"] = data["close"].diff()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out.loc[out["previous_close"].isna(), "true_range"] = np.nan
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["up_range"] = out["true_range"].where(out["change"] > 0, 0.0)
        out["down_range"] = out["true_range"].where(out["change"] < 0, 0.0)
        out.loc[out["change"].isna(), ["up_range", "down_range"]] = np.nan
        out["up_pressure"] = out["up_range"].rolling(params.window, min_periods=params.window).mean()
        out["down_pressure"] = out["down_range"].rolling(params.window, min_periods=params.window).mean()
        out["leadership"] = out["up_pressure"] - out["down_pressure"]
        out["previous_leadership"] = out["leadership"].shift(1)
        out["previous_atr"] = out["atr"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["entry"] = (
            (out["leadership"] > 0)
            & (out["previous_leadership"] <= 0)
            & (out["atr"] < out["previous_atr"])
            & (out["atr"] > 0)
            & (data["close"] > out["previous_high"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeHandoffParams) -> SignalFrame:
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        armed = False
        entry_price = 0.0
        stop = 0.0
        peak = 0.0
        age = 0

        for i in range(len(data)):
            # A previous close's entry decision fills at this bar's open.
            if not active and i > 0 and decisions[i - 1] == 1:
                if not np.isfinite(opens[i]) or opens[i] <= 0:
                    continue
                if not np.isfinite(atrs[i - 1]) or atrs[i - 1] <= 0:
                    continue
                active = True
                armed = False
                age = 0
                entry_price = opens[i]
                peak = entry_price
                stop = entry_price - params.trail_atr * atrs[i - 1]

            if active:
                age += 1
                close = closes[i]
                if not np.isfinite(close) or close <= 0:
                    active = False
                    continue
                breached = close <= stop
                peak = max(peak, close)
                # Close-based activation avoids assuming an intrabar price path.
                if close >= entry_price * 1.005:
                    armed = True
                if armed:
                    stop = max(stop, entry_price)
                    if np.isfinite(atrs[i]) and atrs[i] > 0:
                        stop = max(stop, peak - params.trail_atr * atrs[i])
                if breached or close <= stop or age >= 2:
                    active = False
                else:
                    decisions[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                if np.isfinite(atrs[i]) and atrs[i] > 0:
                    decisions[i] = 1

        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0,
                         "close": [100.0, 101.0, 101.5, 100.0, 100.0],
                         "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"atr": 1.0,
                               "entry": [True, False, False, False, False]}, index=index)
    params = RangeHandoffParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], "open"] = 105.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 0, 0]
    assert (result.data["size"] > 0).all()
