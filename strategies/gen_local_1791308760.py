from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    lookback: int = 30
    stop_atr: float = 1.2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791308760"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.lookback + 3, 15)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if params.lookback < 5 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("lookback must be >= 5 and stop_atr finite and positive")
        out = pd.DataFrame(index=data.index)
        out["valid_close"] = data["close"].where(
            np.isfinite(data["close"]) & (data["close"] > 0)
        )
        out["previous_close"] = out["valid_close"].shift(1)
        out["roc"] = out["valid_close"].pct_change(fill_method=None)
        out["acceleration"] = out["roc"].diff()
        out["lag_acceleration"] = out["acceleration"].shift(1)
        out["acceleration_acf"] = (
            out["acceleration"].rolling(params.lookback).corr(out["lag_acceleration"])
            .clip(-1.0, 1.0)
        )
        out["forecast_return"] = (
            out["roc"] + out["acceleration_acf"] * out["acceleration"]
        )
        out["range"] = data["high"] - data["low"]
        out["high_gap"] = (data["high"] - out["previous_close"]).abs()
        out["low_gap"] = (data["low"] - out["previous_close"]).abs()
        out["true_range"] = out[["range", "high_gap", "low_gap"]].max(
            axis=1, skipna=False
        )
        out["atr"] = out["true_range"].rolling(14).mean()
        out["noise_floor"] = 0.1 * out["atr"] / out["valid_close"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        valid = np.isfinite(
            indicators[["roc", "acceleration", "acceleration_acf",
                        "forecast_return", "atr", "noise_floor"]]
        ).all(axis=1)
        eligible = (
            valid
            & (indicators["atr"] > 0)
            & (indicators["acceleration_acf"] < -0.6)
            & (indicators["roc"] * indicators["acceleration"] > 0)
            & (indicators["roc"] * indicators["forecast_return"] < 0)
            & (indicators["forecast_return"].abs() > indicators["noise_floor"])
        )
        candidates = np.sign(
            indicators["forecast_return"].where(eligible, 0.0)
        ).astype(int).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        direction = 0
        entry_bar = -1
        stop_price = np.nan

        for i in range(len(data)):
            if direction:
                if i == entry_bar:
                    # The prior close scheduled entry; this open is now observable.
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        direction = 0
                        continue
                    # Freeze ATR from the decision bar; never ratchet the stop.
                    stop_price = opens[i] - direction * params.stop_atr * atr[i - 1]
                stopped = (
                    not np.isfinite(closes[i])
                    or direction * (closes[i] - stop_price) < 0
                )
                if stopped or i - entry_bar + 1 >= 2:
                    direction = 0
                raw[i] = direction
                # Exit decisions stay flat for this bar, preventing immediate reentry.
                continue
            if candidates[i] and i + 1 < len(data):
                direction = int(candidates[i])
                entry_bar = i + 1
                raw[i] = direction

        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0,
         "close": 100.0, "volume": 1000.0}, index=index
    )
    features = pd.DataFrame(
        {"roc": 0.0, "acceleration": 0.0, "acceleration_acf": -0.8,
         "forecast_return": 0.0, "atr": 1.0, "noise_floor": 0.001},
        index=index
    )
    features.loc[index[0], ["roc", "acceleration", "forecast_return"]] = (
        -0.01, -0.03, 0.014
    )
    features.loc[index[3], ["roc", "acceleration", "forecast_return"]] = (
        0.01, 0.03, -0.014
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=33)
    def signals(frame):
        return strategy.generate_signals(frame, features, ctx, params).data["signal"].tolist()
    assert signals(data) == [0, 1, 1, 0, -1, -1, 0]
    data.loc[index[1], "close"] = 97.0
    data.loc[index[4], "close"] = 103.0
    assert signals(data) == [0, 1, 0, 0, -1, 0, 0]
    data.loc[index[1], ["open", "close"]] = [120.0, 119.0]
    assert signals(data) == [0, 1, 1, 0, -1, 0, 0]
    flat = data.copy()
    flat[["open", "high", "low", "close"]] = 100.0
    derived = strategy.indicators(flat, params)
    result = strategy.generate_signals(flat, derived, ctx, params).data
    assert result["signal"].eq(0).all() and result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
