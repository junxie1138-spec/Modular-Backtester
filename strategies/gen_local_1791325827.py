from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    frac_order: float = 0.4
    frac_window: int = 64
    noise_window: int = 20
    release_threshold: float = 0.8
    hold_bars: int = 18

    def __post_init__(self):
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError("frac_order must lie between zero and one")
        if self.frac_window < 2 or self.noise_window < 2 or self.hold_bars < 1:
            raise ValueError("Invalid lookback or holding period")
        if not np.isfinite(self.release_threshold) or self.release_threshold <= 0:
            raise ValueError("release_threshold must be finite and positive")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791325827"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(200, params.frac_window + params.noise_window + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.frac_window):
            weight *= (lag - 1 - params.frac_order) / lag
            fractional = fractional + weight * close.shift(lag)
        out["fractional_close"] = fractional
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        quarter = pd.Series(
            data.index.year * 4 + (data.index.month - 1) // 3,
            index=data.index,
        )
        out["quarter"] = quarter
        out["quarter_age"] = quarter.groupby(quarter, sort=False).cumcount()
        out["quarter_anchor"] = fractional.shift(1).where(
            quarter.ne(quarter.shift(1))
        ).ffill()
        out["noise"] = fractional.diff().rolling(
            params.noise_window, min_periods=params.noise_window
        ).std(ddof=0).shift(1)
        denominator = out["noise"] * np.sqrt(out["quarter_age"] + 1.0)
        out["release_score"] = (
            (fractional - out["quarter_anchor"]) / denominator.where(denominator > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["entry_ready"] = (
            (out["quarter_age"] < 5)
            & (out["release_score"] >= params.release_threshold)
            & (close > out["ma200"])
            & np.isfinite(close)
            & np.isfinite(out["release_score"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        eligible = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar < exit_bar:
                    signal[bar] = 1
                else:
                    exit_bar = -1
                continue
            if eligible[bar]:
                signal[bar] = 1
                exit_bar = bar + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = Params(hold_bars=18)
    index = pd.date_range("2020-01-01", periods=24, freq="B")
    data = pd.DataFrame({"close": 100.0}, index=index)
    indicators = pd.DataFrame({"entry_ready": False}, index=index)
    indicators.iloc[0, 0] = True
    result = GeneratedStrategy().generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0] + [1] * 18 + [0] * 5
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()
