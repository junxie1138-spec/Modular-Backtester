from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapTransmissionParams:
    transmission_lookback: int = 20
    atr_lookback: int = 14
    regime_threshold: float = 0.15
    gap_threshold_atr: float = 0.20
    trail_atr: float = 2.0
    hold_days: int = 4


class GeneratedStrategy(BaseStrategy[GapTransmissionParams]):
    strategy_id = "gen_local-w2_1791316308"

    @classmethod
    def params_type(cls) -> type[GapTransmissionParams]:
        return GapTransmissionParams

    @staticmethod
    def warmup_bars(params: GapTransmissionParams) -> int:
        return int(max(params.transmission_lookback + 3, params.atr_lookback + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapTransmissionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        prior = data["close"].shift(1).replace(0.0, np.nan)
        opening = data["open"].replace(0.0, np.nan)
        out["gap_return"] = (data["open"] - prior) / prior
        out["intraday_return"] = (data["close"] - opening) / opening
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior).abs(),
            (data["low"] - prior).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_lookback).mean()
        out["prior_atr"] = out["atr"].shift(1)
        out["gap_atr"] = (data["open"] - prior) / out["prior_atr"].replace(0.0, np.nan)
        length = params.transmission_lookback
        out["cross_moment"] = (out["gap_return"] * out["intraday_return"]).rolling(length).mean().shift(1)
        out["gap_energy"] = out["gap_return"].pow(2).rolling(length).mean().shift(1)
        out["intraday_energy"] = out["intraday_return"].pow(2).rolling(length).mean().shift(1)
        out["transmission"] = out["cross_moment"] / np.sqrt(
            out["gap_energy"] * out["intraday_energy"]
        ).replace(0.0, np.nan)
        out["regime"] = np.select([
            out["transmission"] > params.regime_threshold,
            out["transmission"] < -params.regime_threshold,
        ], [1, -1], default=0).astype(int)
        continuation = (
            out["regime"].eq(1)
            & out["gap_atr"].ge(params.gap_threshold_atr)
            & out["intraday_return"].gt(0.0)
        )
        repair = (
            out["regime"].eq(-1)
            & out["gap_atr"].le(-params.gap_threshold_atr)
            & data["close"].gt(prior)
        )
        ready = np.isfinite(out[["atr", "gap_atr", "transmission"]]).all(axis=1) & out["atr"].gt(0.0)
        out["setup"] = ((continuation | repair) & ready).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapTransmissionParams,
    ) -> SignalFrame:
        if params.trail_atr <= 0.0 or not 3 <= params.hold_days <= 5:
            raise ValueError("trail_atr must be positive and hold_days must be between 3 and 5")
        confirmed = (
            indicators["setup"].fillna(False)
            & indicators["setup"].shift(1).fillna(False)
            & indicators["regime"].eq(indicators["regime"].shift(1))
            & indicators["regime"].ne(0)
        ).to_numpy(dtype=bool)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = np.isfinite(closes[i]) and np.isfinite(atrs[i]) and atrs[i] > 0.0
            if active:
                held += 1
                if valid:
                    # Entry fills this bar when held == 1; exclude the pre-entry close.
                    high_water = max(high_water, closes[i])
                    stop = max(stop, high_water - params.trail_atr * atrs[i])
                if not valid or closes[i] <= stop or held >= params.hold_days:
                    active = False
            elif confirmed[i] and valid:
                active = True
                held = 0
                high_water = -np.inf
                stop = -np.inf
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = GapTransmissionParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100.0, 100.0, 101.0, 104.0, 101.0, 101.0, 101.0]}, index=index)
    features = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0],
        "setup": [True, True, False, False, False, False, False],
        "regime": [1, 1, 1, 1, 1, 1, 1],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    features.loc[index[1], "setup"] = False
    assert GeneratedStrategy.generate_signals(data, features, None, params).data["signal"].eq(0).all()
    features.loc[index[1], "setup"] = True
    data["close"] = [100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    features["atr"] = 1.0
    assert GeneratedStrategy.generate_signals(data, features, None, params).data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0]


if __name__ == "__main__":
    _self_check()
