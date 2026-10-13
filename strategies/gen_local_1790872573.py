from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_window: int = 12
    position_window: int = 20
    atr_window: int = 14
    innovation_threshold: float = 0.025
    max_relative_position: float = 0.35
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790872573"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(200, params.atr_window + 1,
                       max(params.trend_window, params.atr_window + 1)
                       + params.position_window - 1,
                       params.trend_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.trend_window < 2 or params.position_window < 2
                or params.atr_window < 1 or not 3 <= params.hold_bars <= 5
                or not 0 < params.max_relative_position < 1
                or not np.isfinite(params.stop_atr) or params.stop_atr <= 0
                or not np.isfinite(params.innovation_threshold)
                or params.innovation_threshold <= 0):
            raise ValueError("Invalid strategy parameters")
        close = data["close"].astype(float)
        previous = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        out = pd.DataFrame(index=data.index)
        out["atr"] = true_range.rolling(params.atr_window,
                                       min_periods=params.atr_window).mean()
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        n = params.trend_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        out["slope"] = np.nan
        if len(data) >= n:
            out.iloc[n - 1:, out.columns.get_loc("slope")] = np.convolve(
                close.to_numpy(), weights[::-1], mode="valid"
            ) / np.dot(weights, weights)
        out["latent_level"] = (
            close.rolling(n, min_periods=n).mean()
            + out["slope"] * (n - 1) / 2.0
        )
        atr = out["atr"].where(out["atr"] > 0)
        out["residual"] = (close - out["latent_level"]) / atr
        out["innovation"] = out["slope"].diff() / atr.shift(1)
        out["residual_low"] = out["residual"].rolling(
            params.position_window, min_periods=params.position_window
        ).min()
        out["residual_high"] = out["residual"].rolling(
            params.position_window, min_periods=params.position_window
        ).max()
        width = out["residual_high"] - out["residual_low"]
        out["relative_position"] = (
            (out["residual"] - out["residual_low"]) / width.where(width > 0)
        )
        out = out.replace([np.inf, -np.inf], np.nan)
        ready = out.notna().all(axis=1)
        out["entry"] = (
            ready & (close > out["ma200"]) & (out["slope"] > 0)
            & (out["innovation"] > params.innovation_threshold)
            & (out["residual"] < 0)
            & (out["relative_position"] <= params.max_relative_position)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = (data["close"] > indicators["ma200"]).fillna(False).to_numpy()
        entry = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        held = 0
        stop = np.nan
        for i in range(len(data)):
            if pending:
                # Today's open fills yesterday's decision; freeze yesterday's ATR.
                stop = opening[i] - params.stop_atr * atr[i - 1]
                active = True
                pending = False
                held = 0
            if active:
                held += 1
                if close[i] < stop or held >= params.hold_bars or not regime[i]:
                    active = False
                else:
                    signal[i] = 1
            elif entry[i]:
                signal[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = StrategyParams(stop_atr=1.0)
    index = pd.date_range("2020-01-01", periods=7, freq="D")
    data = pd.DataFrame({
        "open": 100.0, "high": 112.0, "low": 96.0,
        "close": [110.0, 110.0, 101.0, 97.0, 100.0, 100.0, 100.0],
        "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [2.0, 2.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        "ma200": 90.0,
        "entry": [True, False, False, False, False, False, False],
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    actual = strategy.generate_signals(data, indicators, ctx, params).data
    assert actual["signal"].tolist() == [0, 1, 1, 1, 0, 0, 0]
    data["close"] = 100.0
    actual = strategy.generate_signals(data, indicators, ctx, params).data
    assert actual["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert strategy.indicators(data, params)["entry"].eq(False).all()
