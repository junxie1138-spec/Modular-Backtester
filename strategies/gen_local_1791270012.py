from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapCompressionParams:
    range_window: int = 20
    atr_window: int = 14
    compression_ratio: float = 0.65
    min_gap_atr: float = 0.15
    trail_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[GapCompressionParams]):
    strategy_id = "gen_local_1791270012"

    @classmethod
    def params_type(cls) -> type[GapCompressionParams]:
        return GapCompressionParams

    def warmup_bars(self, params: GapCompressionParams) -> int:
        return int(max(params.range_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: GapCompressionParams) -> pd.DataFrame:
        if params.range_window < 2 or params.atr_window < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not 0 < params.compression_ratio < 1:
            raise ValueError("compression_ratio must lie between zero and one")
        if not np.isfinite(params.min_gap_atr) or params.min_gap_atr <= 0:
            raise ValueError("min_gap_atr must be finite and positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")

        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["gap"] = data["open"] - out["prior_close"]
        out["intraday"] = data["close"] - data["open"]
        out["true_range"] = pd.concat([
            out["range"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["baseline_range"] = out["range"].rolling(
            params.range_window, min_periods=params.range_window
        ).mean().shift(1)
        out["compression"] = out["range"] / out["baseline_range"].replace(0, np.nan)
        out["gap_atr"] = out["gap"].abs() / out["prior_atr"].replace(0, np.nan)
        out["fade_fraction"] = out["intraday"].abs() / out["gap"].abs().replace(0, np.nan)

        finite = np.isfinite(out).all(axis=1)
        qualifies = (
            finite
            & (out["atr"] > 0)
            & (out["prior_atr"] > 0)
            & (out["range"] > 0)
            & (out["baseline_range"] > 0)
            & (out["compression"] <= params.compression_ratio)
            & (out["gap_atr"] >= params.min_gap_atr)
            & (out["gap"] * out["intraday"] < 0)
            & (out["fade_fraction"] <= 0.5)
        )
        out["setup"] = np.where(
            qualifies, np.sign(out["gap"].fillna(0)), 0
        ).astype(int)
        confirmed = (
            out["setup"].ne(0)
            & out["setup"].eq(out["setup"].shift(1))
            & out["gap"].abs().gt(out["gap"].shift(1).abs())
            & out["range"].lt(out["range"].shift(1))
        )
        out["candidate"] = out["setup"].where(confirmed, 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapCompressionParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        candidate = indicators["candidate"].fillna(0).to_numpy(dtype=int)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        held = 0
        high_water = np.nan
        stop = -np.inf

        for i in range(len(data)):
            valid = np.isfinite(close[i]) and np.isfinite(atr[i]) and atr[i] > 0
            if position:
                held += 1
                if not valid:
                    position = 0
                else:
                    # Start tracking closes only after the next-open entry.
                    high_water = close[i] if not np.isfinite(high_water) else max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                    # The prescribed close-high exit applies to both directions.
                    if close[i] < stop or held >= 2:
                        position = 0
            elif valid and candidate[i] != 0:
                position = int(candidate[i])
                held = 0
                high_water = np.nan
                stop = -np.inf
            decisions[i] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"close": [100, 105, 103, 100, 99, 99, 99]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 7,
        "candidate": [1, 0, 0, -1, 0, 0, 0],
    }, index=index)
    result = GeneratedStrategy().generate_signals(
        data, indicators, None, GapCompressionParams()
    ).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
