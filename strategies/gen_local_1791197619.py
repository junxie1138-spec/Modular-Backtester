from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeEchoParams:
    lookback: int = 36
    atr_multiple: float = 2.0


class GeneratedStrategy(BaseStrategy[RangeEchoParams]):
    strategy_id = "gen_local_1791197619"

    @classmethod
    def params_type(cls):
        return RangeEchoParams

    def warmup_bars(self, params: RangeEchoParams) -> int:
        if params.lookback < 5 or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("lookback must be >= 5 and atr_multiple finite and positive")
        return max(params.lookback + 3, 14)

    def indicators(self, data: pd.DataFrame, params: RangeEchoParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        width = (data["high"] - data["low"]).where(lambda x: x > 0)
        out["log_range"] = np.log(width / data["close"].where(data["close"] > 0))
        out["log_range"] = out["log_range"].replace([np.inf, -np.inf], np.nan)
        window = out["log_range"].rolling(params.lookback, min_periods=params.lookback)
        out["range_mean"] = window.mean()
        out["range_std"] = window.std()
        out["echo_correlation"] = window.corr(out["log_range"].shift(3)).clip(-1.0, 1.0)
        out["echo_correlation"] = out["echo_correlation"].replace([np.inf, -np.inf], np.nan)
        out["forecast"] = out["range_mean"] + out["echo_correlation"] * (
            out["log_range"].shift(2) - out["range_mean"]
        )
        out["close_location"] = (data["close"] - data["low"]) / width
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry"] = (
            (out["echo_correlation"] > 0.25)
            & (out["range_std"] > 0)
            & (out["log_range"] < out["range_mean"])
            & (out["forecast"] > out["log_range"] + 0.25 * out["range_std"])
            & (out["close_location"] > 0.65)
            & (data["close"] > previous_close)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeEchoParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0
            if active:
                # Entry fills at this bar's open; only subsequent closes set the high-water mark.
                age += 1
                if valid:
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.atr_multiple * atrs[i])
                if not valid or closes[i] <= stop or age >= 4:
                    active = False
            elif valid and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=11)
    data = pd.DataFrame({"close": [100, 102, 101, 100, 100, 100, 100, 100, 100, 100, 100]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 5, 5, 1, 1, 1, 1, 1, 1, 1],
        "entry": [True, False, False, False, False, True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, RangeEchoParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
