from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    trend_window: int = 12
    vol_window: int = 20
    atr_window: int = 14
    innovation_threshold: float = 0.4
    stop_atr: float = 1.5
    target_vol: float = 0.12
    max_hold: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1791201868"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.trend_window, params.vol_window, params.atr_window) + 2)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not (params.trend_window >= 3 and params.vol_window >= 2
                and params.atr_window >= 1 and 3 <= params.max_hold <= 5
                and params.innovation_threshold >= 0 and params.stop_atr > 0
                and params.target_vol > 0):
            raise ValueError("Invalid strategy parameters")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        out["log_close"] = np.log(close)
        out["returns"] = out["log_close"].diff()
        n = params.trend_window
        weights = np.arange(n, dtype=float) - (n - 1) / 2.0
        weights /= np.dot(weights, weights)
        out["slope"] = np.nan
        if len(data) >= n:
            out.iloc[n - 1:, out.columns.get_loc("slope")] = np.convolve(
                out["log_close"].to_numpy(), weights[::-1], mode="valid"
            )
        out["slope_change"] = out["slope"].diff()
        out["volatility"] = out["returns"].rolling(params.vol_window).std(ddof=0)
        out["innovation_scale"] = out["volatility"] * np.sqrt(np.dot(weights, weights))
        out["innovation"] = out["slope_change"] / out["innovation_scale"].where(
            out["innovation_scale"] > 0
        )
        previous = close.shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous).abs(),
             (data["low"] - previous).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["candidate"] = (
            (out["slope"] > 0)
            & (out["innovation"] > params.innovation_threshold)
        ).fillna(False)
        out["confirmed"] = (
            out["candidate"] & out["candidate"].shift(1, fill_value=False)
            & (out["innovation"] < out["innovation"].shift(1))
        ).fillna(False)
        out["target_size"] = (
            params.target_vol / (out["volatility"] * np.sqrt(252.0)).where(
                out["volatility"] > 0
            )
        ).replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.95}, index=data.index)
        active = False
        held = 0
        high_water = -np.inf
        stop = -np.inf
        position_size = 0.95
        close_values = data["close"].to_numpy(dtype=float)
        atr_values = indicators["atr"].to_numpy(dtype=float)
        sizes = indicators["target_size"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        exposures = np.full(len(data), 0.95, dtype=float)
        for i in range(len(data)):
            close, atr = close_values[i], atr_values[i]
            valid = np.isfinite(close) and close > 0 and np.isfinite(atr) and atr > 0
            if active:
                held += 1
                if valid:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.stop_atr * atr)
                if not valid or close < stop or held >= params.max_hold:
                    active = False
                else:
                    signals[i] = 1
                    exposures[i] = position_size
            elif valid and confirmed[i] and np.isfinite(sizes[i]) and sizes[i] > 0:
                active = True
                held = 0
                # Entry fills next bar; only subsequent closes belong to the trade.
                high_water = -np.inf
                stop = -np.inf
                position_size = float(sizes[i])
                signals[i] = 1
                exposures[i] = position_size
        df["signal"] = signals
        df["size"] = exposures
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"close": [100, 100, 100, 104, 103, 102, 101, 100]}, index=index)
    ind = pd.DataFrame({
        "atr": [1, 1, 1, 1, 4, 4, 1, 1],
        "target_size": 0.6,
        "confirmed": [False, False, True, False, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, ind, None, StrategyParams(max_hold=5)).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0]
    assert (result["size"] > 0).all()
    assert result["size"].iloc[3] == 0.6
    data["close"] = 100.0
    ind["atr"] = 1.0
    result = strategy.generate_signals(data, ind, None, StrategyParams(max_hold=3)).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 0, 0]


if __name__ == "__main__":
    _self_check()
