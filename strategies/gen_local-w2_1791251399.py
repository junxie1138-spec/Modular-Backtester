from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    frac_window: int = 32
    fraction: float = 0.6
    z_window: int = 40
    atr_window: int = 14
    z_entry: float = 1.5
    body_entry: float = 0.35
    stop_atr: float = 0.8
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791251399"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.frac_window + params.z_window, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (params.frac_window >= 2 and params.z_window >= 2
                and params.atr_window >= 1 and params.max_hold in (1, 2)):
            raise ValueError("Invalid lookback or holding horizon")
        values = [params.fraction, params.z_entry, params.body_entry, params.stop_atr]
        if not (np.isfinite(values).all() and 0 < params.fraction < 1
                and params.z_entry > 0 and 0 < params.body_entry < 1
                and params.stop_atr > 0):
            raise ValueError("Invalid indicator or stop parameters")
        close = data["close"].astype(float)
        out = pd.DataFrame(index=data.index)
        # Fixed finite fractional-difference kernel, newest observation first.
        weights = np.ones(params.frac_window)
        for lag in range(1, params.frac_window):
            weights[lag] = weights[lag - 1] * (lag - 1 - params.fraction) / lag
        fractional = close * weights[0]
        for lag in range(1, params.frac_window):
            fractional = fractional + weights[lag] * close.shift(lag)
        out["fractional_close"] = fractional
        reference = fractional.shift(1).rolling(params.z_window)
        out["fractional_mean"] = reference.mean()
        out["fractional_std"] = reference.std(ddof=0)
        out["fractional_z"] = ((fractional - out["fractional_mean"])
                               / out["fractional_std"].replace(0, np.nan))
        out["candle_range"] = (data["high"] - data["low"]).replace(0, np.nan)
        out["body_balance"] = (close - data["open"]) / out["candle_range"]
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - close.shift(1)).abs(),
             (data["low"] - close.shift(1)).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        z = indicators["fractional_z"].to_numpy(dtype=float)
        body = indicators["body_balance"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        entries = np.select(
            [(z <= -params.z_entry) & (body >= params.body_entry),
             (z >= params.z_entry) & (body <= -params.body_entry)],
            [1, -1], default=0,
        )
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        watermark, stop = np.nan, np.nan
        for i in range(len(data)):
            close = closes[i]
            valid = np.isfinite([close, z[i], body[i], atr[i]]).all() and atr[i] > 0
            if not valid:
                position, age = 0, 0
                watermark, stop = np.nan, np.nan
                continue
            if position:
                # This bar is held because the previous decision fills at its open.
                age += 1
                if position == 1:
                    watermark = close if age == 1 else max(watermark, close)
                    candidate = watermark - params.stop_atr * atr[i]
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = close <= stop
                    reverted = z[i] >= 0
                else:
                    # Symmetric short trail: lowest close, stop only ratchets down.
                    watermark = close if age == 1 else min(watermark, close)
                    candidate = watermark + params.stop_atr * atr[i]
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = close >= stop
                    reverted = z[i] <= 0
                if stopped or reverted or age >= params.max_hold:
                    position, age = 0, 0
                    watermark, stop = np.nan, np.nan
                # Exits cannot re-enter on the same decision bar.
            else:
                position = int(entries[i])
                age = 0
                watermark, stop = np.nan, np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=100)
    close = pd.Series(100 + np.sin(np.arange(100)) + np.arange(100) * 0.1, index=index)
    data = pd.DataFrame({"open": close - 0.5, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 1000.0})
    derived = strategy.indicators(data, params)
    assert derived["fractional_z"].first_valid_index() == index[strategy.warmup_bars(params) - 1]
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    short_data = data.iloc[:8].copy()
    manual = pd.DataFrame({"fractional_z": [-2, -1, -1, 2, 1, 1, -2, 0],
                           "body_balance": [0.5, 0, 0, -0.5, 0, 0, 0.5, 0],
                           "atr": 1.0}, index=short_data.index)
    result = strategy.generate_signals(short_data, manual, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 1]
    manual.iloc[0, 1] = 0
    assert strategy.generate_signals(short_data, manual, ctx, params).data["signal"].iloc[1] == 0
    full = strategy.generate_signals(data, derived, ctx, params).data
    prefix = strategy.generate_signals(data.iloc[:85], derived.iloc[:85], ctx, params).data
    pd.testing.assert_frame_equal(full.iloc[:85], prefix)
    assert full["signal"].isin([-1, 0, 1]).all() and full["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()
