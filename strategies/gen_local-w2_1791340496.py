from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeRecoilParams:
    range_window: int = 20
    autocorr_threshold: float = -0.2
    atr_period: int = 14
    trail_atr: float = 1.6
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[RangeRecoilParams]):
    strategy_id = "gen_local-w2_1791340496"

    @classmethod
    def params_type(cls):
        return RangeRecoilParams

    def warmup_bars(self, params: RangeRecoilParams) -> int:
        return int(max(200, params.range_window + 3, params.atr_period + 1))

    def indicators(self, data: pd.DataFrame, params: RangeRecoilParams) -> pd.DataFrame:
        if (params.range_window < 3 or params.atr_period < 1
                or params.max_hold not in (3, 4, 5)
                or not np.isfinite(params.trail_atr) or params.trail_atr <= 0
                or not -1.0 <= params.autocorr_threshold < 0.0):
            raise ValueError("Invalid range-recoil parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1).where(close.shift(1) > 0)
        out["range"] = data["high"] - data["low"]
        out["normalized_range"] = out["range"] / previous
        out["range_change"] = out["normalized_range"].diff()
        out["lagged_change"] = out["range_change"].shift(1)
        y = out["range_change"]
        x = out["lagged_change"]
        window = params.range_window
        out["mean_change"] = y.rolling(window).mean()
        out["mean_lagged_change"] = x.rolling(window).mean()
        out["lag_variance"] = x.rolling(window).var()
        out["lag_covariance"] = y.rolling(window).cov(x)
        out["autocorrelation"] = y.rolling(window).corr(x)
        out["beta"] = out["lag_covariance"] / out["lag_variance"].where(out["lag_variance"] > 0)
        out["forecast_change"] = out["mean_change"] + out["beta"] * (y - out["mean_lagged_change"])
        out["close_location"] = (close - data["low"]) / out["range"].where(out["range"] > 0)
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        out["atr"] = out["true_range"].rolling(params.atr_period).mean()
        out["ma200"] = close.rolling(200).mean()
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (close > out["ma200"])
            & (close > data["open"])
            & (out["close_location"] >= 0.7)
            & (out["autocorrelation"] <= params.autocorr_threshold)
            & (out["range_change"] > 0)
            & (out["forecast_change"] < 0)
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: RangeRecoilParams,
    ) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        averages = indicators["ma200"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            close, atr, average = closes[i], atrs[i], averages[i]
            valid = np.isfinite(close) and np.isfinite(atr) and np.isfinite(average) and atr > 0
            if active:
                age += 1
                if valid:
                    # Entry fills this bar or earlier; use only in-trade closes.
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trail_atr * atr)
                if not valid or close <= stop or close <= average or age >= params.max_hold:
                    active = False
                    age = 0
                    high_water = stop = -np.inf
            elif valid and close > average and entries[i]:
                active = True
                age = 0
                high_water = stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100.0, 100.0, 102.0, 101.0, 100.0, 100.0, 100.0]}, index=index)
    indicators = pd.DataFrame({"atr": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0], "ma200": 90.0, "entry": [True, False, False, False, False, False, False]}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, RangeRecoilParams(max_hold=5)).data
    # Rising ATR must not loosen the stop established at the 102 close.
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    data["close"] = 100.0
    indicators["atr"] = 1.0
    result = strategy.generate_signals(data, indicators, ctx, RangeRecoilParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
