from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class EchoParams:
    echo_window: int = 40
    echo_threshold: float = 0.20
    min_gap_atr: float = 0.10
    atr_window: int = 14
    trail_atr: float = 1.80
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[EchoParams]):
    strategy_id = "gen_local-w2_1791344910"

    @classmethod
    def params_type(cls):
        return EchoParams

    def warmup_bars(self, params: EchoParams) -> int:
        return int(max(200, params.echo_window + 3, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: EchoParams) -> pd.DataFrame:
        if params.echo_window < 5 or params.atr_window < 1:
            raise ValueError("Invalid indicator lookback")
        if not 0.0 < params.echo_threshold < 1.0:
            raise ValueError("echo_threshold must be between zero and one")
        if not np.isfinite(params.min_gap_atr) or params.min_gap_atr < 0.0:
            raise ValueError("min_gap_atr must be finite and nonnegative")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be finite and positive")
        if params.max_hold < 3 or params.max_hold > 5:
            raise ValueError("max_hold must be between three and five")
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).where(lambda s: s > 0.0)
        out["gap"] = data["open"].div(out["prior_close"]).sub(1.0)
        out["gap_lag1"] = out["gap"].shift(1)
        out["gap_lag2"] = out["gap"].shift(2)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1).where(out["prior_close"].notna())
        out["atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean()
        window = params.echo_window
        out["corr01"] = out["gap"].rolling(window).corr(out["gap_lag1"])
        out["corr02"] = out["gap"].rolling(window).corr(out["gap_lag2"])
        out["corr12"] = out["gap_lag1"].rolling(window).corr(out["gap_lag2"])
        out["partial_denominator"] = np.sqrt(
            (1.0 - out["corr01"].pow(2)).clip(lower=0.0)
            * (1.0 - out["corr12"].pow(2)).clip(lower=0.0)
        )
        out["partial_echo"] = (
            out["corr02"] - out["corr01"] * out["corr12"]
        ).div(out["partial_denominator"].where(out["partial_denominator"] > 1e-8)).clip(-1.0, 1.0)
        out["gap_floor"] = params.min_gap_atr * out["atr"] / out["prior_close"]
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (data["close"] > out["ma200"])
            & (out["atr"] > 0.0)
            & (out["partial_echo"] >= params.echo_threshold)
            & (out["corr01"] <= 0.0)
            & (out["gap_lag1"] > 0.0)
            & (out["gap"] < -out["gap_floor"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: EchoParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf
        for i in range(len(data)):
            valid = (
                np.isfinite(close[i]) and close[i] > 0.0
                and np.isfinite(atr[i]) and atr[i] > 0.0
                and np.isfinite(regime[i])
            )
            if active:
                # The pending entry fills at this bar's open; only in-trade closes count.
                age += 1
                if valid:
                    high_water = max(high_water, close[i])
                    stop = max(stop, high_water - params.trail_atr * atr[i])
                if (
                    not valid or close[i] <= stop
                    or close[i] <= regime[i] or age >= params.max_hold
                ):
                    active = False
                    age = 0
                    high_water = -np.inf
                    stop = -np.inf
            elif valid and entry[i] and close[i] > regime[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"close": [150., 102., 101., 99., 103., 104., 105., 106., 107., 108.]}, index=index)
    ind = pd.DataFrame({
        "atr": [1., 1., 10., 10., 1., 1., 1., 1., 1., 1.],
        "ma200": 90.,
        "entry": [True, False, False, False, True, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, ind, None, EchoParams())
    # Checks next-bar entry, exclusion of the pre-entry close, stop ratcheting,
    # and exactly four held sessions on the second trade.
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    assert (result.data["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()
