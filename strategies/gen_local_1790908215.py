from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    window: int = 40
    entry_score: float = 0.45

    def __post_init__(self):
        if self.window < 10 or not np.isfinite(self.entry_score) or self.entry_score <= 0:
            raise ValueError("window must be >= 10 and entry_score must be finite and positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790908215"

    @classmethod
    def params_type(cls):
        return Params

    @staticmethod
    def warmup_bars(params: Params) -> int:
        return 64 + 5 + 5 + params.window

    @staticmethod
    def indicators(data: pd.DataFrame, params: Params) -> pd.DataFrame:
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        frac = close.copy()
        weight = 1.0
        # Fixed 64-tap fractional difference, d=0.5; all taps use current or past closes.
        for lag in range(1, 64):
            weight *= (lag - 1 - 0.5) / lag
            frac = frac + weight * close.shift(lag)
        out = pd.DataFrame(index=data.index)
        out["frac_close"] = frac
        out["smooth_frac"] = frac.rolling(5, min_periods=5).mean()
        out["lagged_frac"] = out["smooth_frac"].shift(5)
        history = out["smooth_frac"].rolling(params.window, min_periods=params.window)
        out["center"] = history.mean()
        out["scale"] = history.std(ddof=0)
        out["autocorrelation"] = history.corr(out["lagged_frac"]).replace(
            [np.inf, -np.inf], np.nan
        ).clip(-1.0, 1.0)
        out["standardized_frac"] = (
            (out["smooth_frac"] - out["center"])
            / out["scale"].where(out["scale"] > 1e-12)
        )
        out["forecast"] = (
            out["autocorrelation"] * out["standardized_frac"]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        score = indicators["forecast"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        candidates = np.where(
            score > params.entry_score, 1,
            np.where(score < -params.entry_score, -1, 0),
        )
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for bar, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Exit only on the opposite entry condition; keep this bar flat.
                position = 0
            positions[bar] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params()
    index = pd.date_range("2020-01-01", periods=240, freq="B")
    t = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(0.001 * t + 0.03 * np.sin(t / 9.0))
    data = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000000.0}, index=index,
    )
    indicators = GeneratedStrategy.indicators(data, params)
    prefix = GeneratedStrategy.indicators(data.iloc[:180], params)
    pd.testing.assert_frame_equal(indicators.iloc[:180], prefix)
    assert indicators["autocorrelation"].first_valid_index() <= index[
        GeneratedStrategy.warmup_bars(params) - 1
    ]
    scores = pd.DataFrame(
        {"forecast": [0.0, 1.0, 0.0, np.nan, -1.0, -1.0, 0.0, 1.0, 1.0]},
        index=index[:9],
    )
    result = GeneratedStrategy.generate_signals(data.iloc[:9], scores, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, -1, -1, 0]
    assert result["size"].gt(0).all()
    constant = data.copy()
    constant["close"] = 100.0
    flat = GeneratedStrategy.indicators(constant, params)
    assert flat["forecast"].eq(0.0).all()
    assert GeneratedStrategy.generate_signals(
        constant, flat, None, params
    ).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()
