from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    signature_window: int = 5
    area_threshold: float = 0.025
    atr_window: int = 14
    stop_atr: float = 1.8
    holding_days: int = 4


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = "gen_local_1791263261"

    @classmethod
    def params_type(cls) -> type[SignatureSeasonParams]:
        return SignatureSeasonParams

    def warmup_bars(self, params: SignatureSeasonParams) -> int:
        return int(max(params.signature_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: SignatureSeasonParams) -> pd.DataFrame:
        if params.signature_window < 2 or params.atr_window < 1:
            raise ValueError("Indicator windows must be positive; signature_window must exceed one.")
        if not 3 <= params.holding_days <= 5:
            raise ValueError("holding_days must be between three and five.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        if not np.isfinite(params.area_threshold) or params.area_threshold < 0:
            raise ValueError("area_threshold must be finite and nonnegative.")

        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        opening = data["open"].where(lambda s: s > 0)
        gap = (opening / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        body = (data["close"] / opening - 1.0).replace([np.inf, -np.inf], np.nan)
        length = params.signature_window
        area = pd.Series(0.0, index=data.index)
        for lag in range(1, length):
            cross = gap.shift(lag) * body - body.shift(lag) * gap
            area = area + cross.rolling(length - lag, min_periods=length - lag).sum()
        variation = (gap.abs() + body.abs()).rolling(length, min_periods=length).sum()
        normalized_area = 0.5 * area / variation.pow(2).where(variation > 0)

        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous_close.notna())
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        weekday = pd.Series(data.index.dayofweek, index=data.index)
        return pd.DataFrame({
            "signature_area": normalized_area,
            "gap_sum": gap.rolling(length, min_periods=length).sum(),
            "body_sum": body.rolling(length, min_periods=length).sum(),
            "atr": atr,
            "seasonal_window": weekday.between(1, 3),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureSeasonParams) -> SignalFrame:
        candidate = (
            indicators["seasonal_window"]
            & (indicators["signature_area"] < -params.area_threshold)
            & (indicators["gap_sum"] < 0)
            & (indicators["body_sum"] > 0)
            & np.isfinite(indicators["atr"])
            & (indicators["atr"] > 0)
            & np.isfinite(data["close"])
            & (data["close"] > 0)
        ).fillna(False)
        confirmed = (candidate & candidate.shift(1, fill_value=False)).to_numpy(dtype=bool)
        opening = data["open"].to_numpy(dtype=float)
        closing = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        desired = np.zeros(len(data), dtype=int)
        active = False
        stop_price = np.nan
        age = 0

        for i in range(len(data)):
            # A decision on the previous close first becomes a position at this open.
            if not active and i > 0 and desired[i - 1] == 1:
                if not np.isfinite(opening[i]) or opening[i] <= 0:
                    raise ValueError("Entry open must be finite and positive.")
                stop_price = opening[i] - params.stop_atr * atr[i - 1]
                active = True
                age = 0

            if active:
                age += 1
                if not np.isfinite(closing[i]):
                    raise ValueError("Close must be finite while holding a position.")
                if closing[i] < stop_price or age >= params.holding_days:
                    active = False
                    desired[i] = 0
                else:
                    desired[i] = 1
            elif confirmed[i]:
                desired[i] = 1

        df = pd.DataFrame({"signal": desired, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="B")
    data = pd.DataFrame({
        "open": [100.0] * 8,
        "high": [101.0] * 8,
        "low": [95.0] * 8,
        "close": [100.0, 100.0, 100.0, 100.0, 97.0, 100.0, 100.0, 100.0],
        "volume": [1000.0] * 8,
    }, index=index)
    features = pd.DataFrame({
        "signature_area": [-0.1, -0.1] + [0.0] * 6,
        "gap_sum": [-0.01] * 8,
        "body_sum": [0.01] * 8,
        "atr": [1.0, 1.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        "seasonal_window": [True] * 8,
    }, index=index)
    strategy = GeneratedStrategy()
    params = SignatureSeasonParams(stop_atr=2.0)
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    result = strategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
