from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureSeasonParams:
    area_threshold: float = 0.12
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[SignatureSeasonParams]):
    strategy_id = "gen_local-w2_1791311875"

    @classmethod
    def params_type(cls):
        return SignatureSeasonParams

    def warmup_bars(self, params):
        return 15

    def indicators(self, data, params):
        if not 0.0 <= params.area_threshold < 1.0:
            raise ValueError("area_threshold must be in [0, 1)")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        gap = np.log(opening / close.shift(1))
        body = np.log(close / opening)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric level-two signature of the five-session gap/body path.
        for older in range(4, 0, -1):
            for newer in range(older):
                area += 0.5 * (
                    gap.shift(older) * body.shift(newer)
                    - body.shift(older) * gap.shift(newer)
                )
        scale = gap.abs().rolling(5, min_periods=5).sum() * body.abs().rolling(5, min_periods=5).sum()
        normalized = (area / scale.where(scale > 0)).replace([np.inf, -np.inf], np.nan)
        previous = close.shift(1)
        tr = pd.concat(
            [data["high"] - data["low"], (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1).where(previous.notna())
        atr = tr.rolling(14, min_periods=14).mean().where(lambda s: s > 0)
        dates = data.index.normalize()
        days_to_end = ((dates + pd.offsets.BMonthEnd(0)) - dates).days
        seasonal = (days_to_end <= 6) | (dates.day <= 3)
        recovery = (body > 0) & (close > previous)
        return pd.DataFrame(
            {"area": normalized, "atr": atr, "seasonal": seasonal,
             "recovery": recovery}, index=data.index
        )

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        entry = (
            indicators["seasonal"].fillna(False).astype(bool)
            & indicators["recovery"].fillna(False).astype(bool)
            & (indicators["area"] < -params.area_threshold)
            & indicators["atr"].notna()
        ).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        high = -np.inf
        stop = -np.inf
        age = 0
        for i in range(len(data)):
            price, atr = closes[i], atrs[i]
            valid = np.isfinite(price) and price > 0 and np.isfinite(atr) and atr > 0
            if active:
                age += 1
                if valid:
                    # Entry executes this bar after the preceding close's decision.
                    high = max(high, price)
                    stop = max(stop, high - params.trail_atr * atr)
                if not valid or price <= stop or age >= 8:
                    active = False
            elif valid and entry[i]:
                active = True
                high = -np.inf
                stop = -np.inf
                age = 0
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.bdate_range("2020-01-01", periods=15)
    data = pd.DataFrame({"close": [100, 101, 104, 103, 101] + [100] * 10}, index=index)
    indicators = pd.DataFrame(
        {"area": [-0.3] * 15, "atr": [1, 1, 1, 10, 10] + [1] * 10,
         "seasonal": [True] + [False] * 4 + [True] + [False] * 9,
         "recovery": True}, index=index
    )
    result = GeneratedStrategy().generate_signals(
        data, indicators, None, SignatureSeasonParams()
    ).data
    # A widening ATR cannot lower the stop; the second trade lasts eight fills.
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0] + [1] * 8 + [0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
