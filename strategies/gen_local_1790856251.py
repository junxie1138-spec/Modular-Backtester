from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapChangeParams:
    regime_bars: int = 5
    percentile_bars: int = 126
    shock_percentile: float = 0.85
    negative_fraction: float = 0.60
    atr_bars: int = 14
    trail_atr: float = 1.25
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GapChangeParams]):
    strategy_id = "gen_local_1790856251"

    @classmethod
    def params_type(cls):
        return GapChangeParams

    @staticmethod
    def warmup_bars(params):
        return int(max(params.regime_bars + params.percentile_bars + 2,
                       params.atr_bars + 1))

    @staticmethod
    def indicators(data, params):
        if params.regime_bars < 2 or params.percentile_bars < 2 or params.atr_bars < 1:
            raise ValueError("Lookback lengths must be positive; regime and percentile windows must be at least two.")
        if not 0 < params.shock_percentile < 1 or not 0 < params.negative_fraction <= 1:
            raise ValueError("Invalid percentile or negative-gap fraction.")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0 or params.max_hold_bars not in (1, 2):
            raise ValueError("Use a positive ATR multiplier and a one- or two-session holding limit.")

        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0, np.nan)
        out["gap"] = (data["open"] / previous_close - 1).replace([np.inf, -np.inf], np.nan)
        out["prior_gap_mean"] = out["gap"].shift(1).rolling(params.regime_bars).mean()
        negative = (out["gap"] < 0).astype(float).where(out["gap"].notna())
        out["prior_negative_fraction"] = negative.shift(1).rolling(params.regime_bars).mean()
        out["gap_innovation"] = out["gap"] - out["prior_gap_mean"]
        out["shock_threshold"] = out["gap_innovation"].shift(1).rolling(params.percentile_bars).quantile(params.shock_percentile)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars).mean()
        out["intraday_return"] = (data["close"] / data["open"].replace(0, np.nan) - 1).replace([np.inf, -np.inf], np.nan)
        ready = np.isfinite(out).all(axis=1) & out["atr"].gt(0)
        out["entry"] = (
            ready
            & out["prior_gap_mean"].lt(0)
            & out["prior_negative_fraction"].ge(params.negative_fraction)
            & out["gap"].gt(0)
            & out["gap_innovation"].gt(out["shock_threshold"])
            & out["intraday_return"].ge(0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data, indicators, ctx, params):
        df = pd.DataFrame(index=data.index)
        decisions = np.zeros(len(data), dtype=int)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        active = False
        held = 0
        highest_close = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                # The entry decision fills at this bar's open; only subsequent closes enter the high-water mark.
                held += 1
                if valid:
                    highest_close = max(highest_close, close)
                    stop = max(stop, highest_close - params.trail_atr * atr)
                if not valid or close <= stop or held >= params.max_hold_bars:
                    active = False
                else:
                    decisions[i] = 1
            elif valid and entries[i]:
                active = True
                held = 0
                highest_close = -np.inf
                stop = -np.inf
                decisions[i] = 1

        df["signal"] = decisions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({"close": [100, 102, 100, 100, 101, 102, 102]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 10, 1, 1, 1, 1],
        "entry": [True, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, GapChangeParams()).data
    # Rising ATR cannot loosen the first trade's stop; the second trade reaches its two-session deadline.
    assert result["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
